#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场状态失败理解层 (Regime Failure Understanding Layer)
=====================================================

从"识别+过滤"升级到"识别+理解+自适应"的认知层。
基于 INDICATOR_DEEP_LOGIC.md 第五章4类Regime失败根因 + 第六章维度独立性原则。

设计哲学:
  - 过滤模式: regime=ranging → 不交易 (浪费机会)
  - 理解模式: regime=ranging → 理解为何trend_following失效
                       → 切换到该regime下有效的策略 (mean_reversion)
                       → 调整SL/TP为该regime的微观结构特征

学术来源:
  - INDICATOR_DEEP_LOGIC.md 第七章"长期(因果理解)"路线图
  - Regime-Switching Strategy (Hamilton 1989 + Kritzman 2012)
  - Bayesian Strategy Selection (Adams-MacKay 2007 变体)
  - Easley 2012 VPIN 毒性流检测

集成位置:
  RegimeDetectionSystem (识别层) → RegimeFailureUnderstanding (理解层) → StrategyFusion (融合层)

版本: v447.0.0
作者: agent_1_coordinator
创建: 2026-06-28
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
from collections import deque
import logging

import numpy as np

logger = logging.getLogger("hermes.regime_failure_understanding")


# ============================================================================
# 1. 失败模式枚举（来自 INDICATOR_DEEP_LOGIC.md 第五章）
# ============================================================================

class FailureMode(Enum):
    """4类已识别的 Regime 失败模式"""
    VOLATILE_FALSE_BREAKOUT = "volatile_false_breakout"        # 高波动假突破
    DOWNTREND_COUNTER_TREND = "downtrend_counter_trend"        # 下跌逆势做多
    RANGING_SIGNAL_NOISE = "ranging_signal_noise"              # 震荡信号噪声
    UPTREND_END_MOMENTUM_DECAY = "uptrend_end_momentum_decay"  # 趋势末端动量衰减


class StrategyParadigm(Enum):
    """策略范式（与 worldview 对应）"""
    TREND_FOLLOWING = "trend_following"            # 趋势跟随: MACD/EMA/ROC
    MEAN_REVERSION = "mean_reversion"              # 均值回归: RSI/Bollinger
    SCALPING = "scalping"                          # 剥头皮: 高频小利
    REDUCED_POSITION = "reduced_position"          # 减仓防守: 仅观察
    COUNTER_TREND_SHORT = "counter_trend_short"    # 顺势做空: 仅 downtrend
    BREAKOUT = "breakout"                          # 突破: 关键位突破


# ============================================================================
# 2. 根因知识库（来自 INDICATOR_DEEP_LOGIC.md 第五、六章）
# ============================================================================

@dataclass
class RegimeFailurePattern:
    """单个失败模式签名"""
    failure_mode: FailureMode
    applicable_regimes: List[str]                       # 适用 regime
    failed_paradigms: List[StrategyParadigm]            # 该 regime 下失效的范式
    effective_paradigms: List[StrategyParadigm]         # 该 regime 下有效的范式
    root_cause: str                                     # 根因
    indicator_failure_predictors: Dict[str, float]      # 指标→失效概率
    recommended_sl_multiplier: float                    # SL 调整倍数
    recommended_tp_multiplier: float                    # TP 调整倍数
    recommended_position_coef: float                    # 仓位调整系数
    academic_source: str                                # 学术来源


# 知识库（硬编码 INDICATOR_DEEP_LOGIC.md 第五章 4 类根因）
REGIME_FAILURE_KNOWLEDGE_BASE: Dict[str, RegimeFailurePattern] = {
    "ranging": RegimeFailurePattern(
        failure_mode=FailureMode.RANGING_SIGNAL_NOISE,
        applicable_regimes=["ranging", "sideways"],
        failed_paradigms=[StrategyParadigm.TREND_FOLLOWING, StrategyParadigm.BREAKOUT],
        effective_paradigms=[StrategyParadigm.MEAN_REVERSION, StrategyParadigm.SCALPING],
        root_cause=(
            "震荡市场无明确方向, 趋势指标(MACD/EMA)在0轴附近反复金叉死叉, "
            "产生大量假信号; 同时假突破频繁, SL被反复扫损后价格回到区间内"
        ),
        indicator_failure_predictors={
            "macd_cross": 0.85,         # 85% 失效概率
            "ema_breakout": 0.80,
            "roc": 0.60,                # 领先指标也部分失效
            "rsi_divergence": 0.20,     # 均值回归指标有效(仅20%失效)
            "bollinger_squeeze": 0.15,
            "willr_signal": 0.40,
            "mfi_divergence": 0.35,
            "volume_anomaly": 0.30,
        },
        recommended_sl_multiplier=0.8,   # SL 收紧到区间宽度 80%
        recommended_tp_multiplier=0.6,   # TP 缩小(区间内空间有限)
        recommended_position_coef=0.8,   # 仓位略降
        academic_source="INDICATOR_DEEP_LOGIC.md L308-L318",
    ),
    "volatile": RegimeFailurePattern(
        failure_mode=FailureMode.VOLATILE_FALSE_BREAKOUT,
        applicable_regimes=["volatile", "high_vol"],
        failed_paradigms=[StrategyParadigm.TREND_FOLLOWING, StrategyParadigm.SCALPING, StrategyParadigm.BREAKOUT],
        effective_paradigms=[StrategyParadigm.REDUCED_POSITION],
        root_cause=(
            "高波动=价格大幅震荡, 容易触发假突破; SL被频繁扫损后才回到预期方向; "
            "MACD等滞后指标频繁金叉死叉, 信号噪声大"
        ),
        indicator_failure_predictors={
            "macd_cross": 0.90,
            "ema_breakout": 0.85,
            "roc": 0.70,
            "rsi_divergence": 0.60,      # 高波动下均值回归也部分失效
            "bollinger_squeeze": 0.50,
            "willr_signal": 0.55,
            "volume_anomaly": 0.30,      # 量价信号在高波动下相对可靠
            "mfi_divergence": 0.45,
        },
        recommended_sl_multiplier=1.5,   # SL 放宽(ATR-based, 避免被扫损)
        recommended_tp_multiplier=1.2,
        recommended_position_coef=0.5,   # 大幅减仓
        academic_source="INDICATOR_DEEP_LOGIC.md L279-L293",
    ),
    "downtrend": RegimeFailurePattern(
        failure_mode=FailureMode.DOWNTREND_COUNTER_TREND,
        applicable_regimes=["downtrend", "bear"],
        failed_paradigms=[StrategyParadigm.TREND_FOLLOWING],   # 做多方向失效
        effective_paradigms=[StrategyParadigm.COUNTER_TREND_SHORT, StrategyParadigm.MEAN_REVERSION],
        root_cause=(
            "下跌趋势中做多=逆势, 即使短期反弹也大概率继续下跌; "
            "反弹通常是死猫跳, 诱多后继续下跌"
        ),
        indicator_failure_predictors={
            "macd_cross": 0.75,         # 金叉多为反弹噪声
            "ema_breakout": 0.70,
            "roc": 0.50,
            "rsi_divergence": 0.30,     # RSI 超卖反弹有时有效
            "bollinger_squeeze": 0.40,
            "willr_signal": 0.35,
            "mfi_divergence": 0.30,
        },
        recommended_sl_multiplier=1.0,
        recommended_tp_multiplier=0.8,
        recommended_position_coef=0.6,
        academic_source="INDICATOR_DEEP_LOGIC.md L295-L306",
    ),
    "uptrend_end": RegimeFailurePattern(
        failure_mode=FailureMode.UPTREND_END_MOMENTUM_DECAY,
        applicable_regimes=["bull"],    # 但需检测动量衰减
        failed_paradigms=[StrategyParadigm.TREND_FOLLOWING, StrategyParadigm.BREAKOUT],
        effective_paradigms=[StrategyParadigm.REDUCED_POSITION, StrategyParadigm.SCALPING],
        root_cause=(
            "趋势末端特征: 价格创新高但动量衰减(MACD柱缩短); "
            "MACD金叉陷阱=趋势末端短暂上穿后立即下穿; 单信号无确认"
        ),
        indicator_failure_predictors={
            "macd_cross": 0.80,         # 趋势末端金叉陷阱
            "ema_breakout": 0.65,
            "roc": 0.45,                # ROC 领先, 但衰减期信号弱
            "rsi_divergence": 0.20,     # 顶背离检测在此场景有效
            "bollinger_squeeze": 0.30,
            "willr_signal": 0.40,
            "mfi_divergence": 0.35,
        },
        recommended_sl_multiplier=1.0,
        recommended_tp_multiplier=0.7,
        recommended_position_coef=0.7,
        academic_source="INDICATOR_DEEP_LOGIC.md L320-L331",
    ),
}


# ============================================================================
# 3. 失败交易记录（动态输入）
# ============================================================================

@dataclass
class FailureTradeRecord:
    """失败交易记录（用于动态学习）"""
    timestamp: float
    regime: str                              # 当时 regime
    strategy_paradigm: StrategyParadigm      # 使用的策略范式
    indicators_used: List[str]               # 使用的指标
    pnl: float                               # 损益
    failure_reason: str                      # 失败原因
    indicators_state: Dict[str, float]       # 当时各指标状态


# ============================================================================
# 4. 辅助数据结构
# ============================================================================

@dataclass
class StrategySwitchRecommendation:
    """策略切换建议"""
    recommended_paradigm: StrategyParadigm
    confidence: float
    reason: str


@dataclass
class FailureUnderstandingReport:
    """失败理解报告（主输出）"""
    effective_regime: str
    detected_failure_mode: Optional[FailureMode]
    root_cause: str
    current_paradigm: StrategyParadigm
    recommended_paradigm: StrategyParadigm
    strategy_switch_confidence: float
    indicator_failure_probs: Dict[str, float]
    failure_warnings: List[Dict[str, Any]]
    recommended_sl_multiplier: float
    recommended_tp_multiplier: float
    recommended_position_coef: float
    counterfactual_win_rate_delta: float
    regime_confidence: float
    is_transitioning: bool

    def should_switch_strategy(self) -> bool:
        """是否应该切换策略"""
        return (
            self.current_paradigm != self.recommended_paradigm
            and self.strategy_switch_confidence > 0.6
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于传递给 strategy_fusion）"""
        return {
            "effective_regime": self.effective_regime,
            "failure_mode": self.detected_failure_mode.value if self.detected_failure_mode else None,
            "root_cause": self.root_cause,
            "current_paradigm": self.current_paradigm.value,
            "recommended_paradigm": self.recommended_paradigm.value,
            "strategy_switch_confidence": self.strategy_switch_confidence,
            "indicator_failure_probs": self.indicator_failure_probs,
            "failure_warnings": self.failure_warnings,
            "sl_multiplier": self.recommended_sl_multiplier,
            "tp_multiplier": self.recommended_tp_multiplier,
            "position_coef": self.recommended_position_coef,
            "counterfactual_delta": self.counterfactual_win_rate_delta,
        }


# ============================================================================
# 4.5 v598 Phase J: FailureModeMatcher — 自动联动 regime + 连续亏损 → FailureMode
# ============================================================================

# regime 别名归一化表 (与 RegimeFailureUnderstanding._detect_uptrend_end 一致)
# v598 Phase J MEDIUM #3 修复: 补充 classify_market_regime 的 "trend"/"quiet" 输出
#   - "trend" (ADX≥25 强趋势) → "trend_strong" (健康强趋势, 非 failure mode)
#   - "quiet" (低波动+趋势弱) → "ranging" (近似震荡)
# v598 Phase J MEDIUM #4 修复: "uptrend" 不再映射到 UPTREND_END_MOMENTUM_DECAY
#   - 健康上升趋势 (uptrend/bull) 不是失败模式, 仅 "uptrend_end" (趋势末端) 才是
#   - 因此保留 "uptrend" 归一化名 (供 detect_failure 匹配), 但从 _REGIME_TO_FAILURE_MODE 移除
_REGIME_NORMALIZE_MAP: Dict[str, str] = {
    "sideways": "ranging",
    "range": "ranging",
    "consolidation": "ranging",
    "ranging": "ranging",
    "quiet": "ranging",  # MEDIUM #3: classify_market_regime 的低波动+弱趋势
    "bull": "uptrend",
    "uptrend": "uptrend",
    "trend": "trend_strong",  # MEDIUM #3: classify_market_regime 的强趋势 (ADX≥25)
    "trend_strong": "trend_strong",
    "bear": "downtrend",
    "downtrend": "downtrend",
    "volatile": "volatile",
    "high_vol": "volatile",
    "low_vol": "ranging",  # 低波动近似 ranging
    "uptrend_end": "uptrend_end",
}


def _normalize_regime_name(regime: str) -> str:
    """归一化 regime 名到知识库大类 (ranging/volatile/downtrend/uptrend/uptrend_end/trend_strong)"""
    if not regime:
        return "ranging"
    regime_lower = str(regime).lower()
    return _REGIME_NORMALIZE_MAP.get(regime_lower, regime_lower)


# regime → FailureMode 匹配表 (基于 REGIME_FAILURE_KNOWLEDGE_BASE 的 4 大根因)
# v598 Phase J MEDIUM #4 修复: 移除 "uptrend" (健康上升趋势非失败模式)
#   - 仅 "uptrend_end" (趋势末端动量衰减) 触发 UPTREND_END_MOMENTUM_DECAY
#   - "trend_strong" (健康强趋势) 不在此表 → 不触发警报 (正确行为)
_REGIME_TO_FAILURE_MODE: Dict[str, "FailureMode"] = {
    "ranging": FailureMode.RANGING_SIGNAL_NOISE,
    "volatile": FailureMode.VOLATILE_FALSE_BREAKOUT,
    "downtrend": FailureMode.DOWNTREND_COUNTER_TREND,
    "uptrend_end": FailureMode.UPTREND_END_MOMENTUM_DECAY,
}


@dataclass
class FailureAlert:
    """v598 Phase J: 连续亏损触发的失败警报 (自动联动产物)"""
    detected_at_trade_idx: int                          # 触发时的 trade 序号
    regime: str                                         # 触发时的 regime (归一化后)
    failure_mode: FailureMode                           # 匹配的失败模式
    consec_loss_count: int                              # 连续亏损笔数
    consec_loss_pnl: float                              # 连续亏损总 PnL
    avg_loss: float                                     # 平均单笔亏损
    recommended_paradigm: StrategyParadigm              # 建议切换的范式
    root_cause_hint: str                                # 根因提示

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "failure_mode": self.failure_mode.value,
            "consec_loss_count": self.consec_loss_count,
            "consec_loss_pnl": round(self.consec_loss_pnl, 4),
            "avg_loss": round(self.avg_loss, 4),
            "recommended_paradigm": self.recommended_paradigm.value,
            "root_cause_hint": self.root_cause_hint,
            "detected_at_trade_idx": self.detected_at_trade_idx,
        }


class FailureModeMatcher:
    """v598 Phase J: 自动联动 regime + 连续亏损 → FailureMode 检测器

    用户铁律:
      "深入理解策略在特定市场状态下失败的根本原因"
      "形成闭环"

    闭环设计:
      1. 每笔 trade 关闭时调用 on_trade_closed(pnl, regime)
      2. 累积滑动窗口 (默认 20 笔)
      3. 检测同 regime 下的连续亏损 (默认 ≥3 笔)
      4. 命中时自动匹配 FailureMode + 推荐 StrategyParadigm
      5. detect_failure() 在 decide() 调用, 若有警报则写入 round_results.warnings

    ERR-110 应用边界:
      - 不新增硬过滤 (不修改入场/出场条件)
      - 仅产出 failure_alert 供 observe/记录, 不直接 block 交易
      - understand() 仍由 _apply_regime_failure_understanding 调用, 此处只做"检测+警报"
    """

    # regime → 推荐切换的范式 (与 REGIME_FAILURE_KNOWLEDGE_BASE.effective_paradigms[0] 一致)
    # v598 Phase J MEDIUM #4: 移除 "uptrend" (健康上升趋势不触发警报, 无需推荐范式)
    #   - "trend_strong" 也不在警报触发表中, 但作为兜底推荐 TREND_FOLLOWING
    _REGIME_RECOMMENDED_PARADIGM: Dict[str, StrategyParadigm] = {
        "ranging": StrategyParadigm.MEAN_REVERSION,
        "volatile": StrategyParadigm.REDUCED_POSITION,
        "downtrend": StrategyParadigm.COUNTER_TREND_SHORT,
        "uptrend_end": StrategyParadigm.REDUCED_POSITION,
        "trend_strong": StrategyParadigm.TREND_FOLLOWING,  # 健康强趋势: 跟随趋势
    }

    # regime → 根因提示 (摘要自 REGIME_FAILURE_KNOWLEDGE_BASE.root_cause)
    # v598 Phase J MEDIUM #4: 移除 "uptrend" (健康上升趋势无根因提示)
    _REGIME_ROOT_CAUSE_HINT: Dict[str, str] = {
        "ranging": "震荡市趋势指标失效, 假突破频繁扫损",
        "volatile": "高波动假突破, SL 频繁被扫",
        "downtrend": "下跌趋势中做多=逆势, 反弹多为死猫跳",
        "uptrend_end": "趋势末端动量衰减, MACD 金叉陷阱",
        "trend_strong": "强趋势中跟随方向, 注意 ADX 衰减信号",
    }

    def __init__(
        self,
        consec_loss_threshold: int = 3,
        window_size: int = 20,
        loss_pnl_threshold: float = 0.0,  # PnL ≤ 此值视为亏损
        alert_cooldown: int = 5,  # v598 Phase J MEDIUM #6: 警报冷却窗口
    ):
        """
        Args:
            consec_loss_threshold: 触发警报的连续亏损笔数 (用户硬约束 max_consec=3)
            window_size: 滑动窗口大小 (只跟踪最近 N 笔 trade)
            loss_pnl_threshold: 亏损判定阈值 (PnL ≤ 此值视为亏损, 默认 0.0)
            alert_cooldown: 警报冷却笔数 (触发后 N 笔 trade 内不再触发新警报)
                v598 Phase J MEDIUM #6 修复: 原 _last_alert_trade_idx 仅阻止同一 trade_idx 重复触发,
                下笔亏损仍会再次触发 (窗口仍包含连续亏损段). 改为 cooldown 窗口机制.
                默认 5 笔: 给策略切换/仓位调整留出观察期, 避免警报风暴
        """
        self.consec_loss_threshold = max(1, consec_loss_threshold)
        self.window_size = max(self.consec_loss_threshold, window_size)
        self.loss_pnl_threshold = loss_pnl_threshold
        self.alert_cooldown = max(0, alert_cooldown)
        # 滑动窗口: [(pnl, regime), ...]
        self._recent_trades: deque = deque(maxlen=self.window_size)
        # 已触发的警报 (冷却窗口: 触发后 alert_cooldown 笔 trade 内不再触发)
        self._last_alert_trade_idx: int = -self.alert_cooldown - 1  # 初始允许立即触发
        self._trade_counter: int = 0
        # 历史警报 (供查询)
        self._alert_history: List[FailureAlert] = []

    def on_trade_closed(self, pnl: float, regime: str) -> Optional[FailureAlert]:
        """记录一笔已关闭的 trade, 自动检测连续亏损

        Args:
            pnl: 该 trade 的实现 PnL
            regime: 该 trade 关闭时的 regime (原始命名, 内部归一化)

        Returns:
            若触发警报则返回 FailureAlert, 否则返回 None
        """
        self._trade_counter += 1
        normalized_regime = _normalize_regime_name(regime)
        self._recent_trades.append((float(pnl), normalized_regime))

        # 检测同 regime 下的连续亏损
        alert = self._detect_consec_loss_alert(normalized_regime)
        if alert is not None:
            self._alert_history.append(alert)
            logger.warning(
                "FailureModeMatcher ALERT: regime=%s, failure_mode=%s, "
                "consec_loss=%d (pnl_total=%.4f, avg=%.4f), recommended=%s",
                alert.regime, alert.failure_mode.value,
                alert.consec_loss_count, alert.consec_loss_pnl, alert.avg_loss,
                alert.recommended_paradigm.value,
            )
        return alert

    def detect_failure(self, current_regime: str) -> Optional[FailureAlert]:
        """在 decide() 调用: 检查当前 regime 下是否有未消解的警报

        与 on_trade_closed 的区别:
          - on_trade_closed: trade 关闭时调用, 自动检测并返回新警报
          - detect_failure: decide() 调用, 查询当前是否有活跃警报

        Args:
            current_regime: 当前 regime (原始命名, 内部归一化)

        Returns:
            若最近一次警报的 regime 与当前 regime 匹配且未消解, 返回该警报
            否则返回 None

        v598 Phase J MEDIUM #5 修复: 跨 regime 误消解
          原 bug: 若最近一笔 trade 是盈利 (无论什么 regime), 警报立即消解
                 → uptrend 警报在 ranging 盈利时被错误消解
          修复: 只有当最近一笔 trade 是盈利 **且** 其 regime 与警报 regime 一致时才消解
                 → 不同 regime 的盈利不影响当前警报 (符合"同 regime 连续亏损"语义)
        """
        if not self._alert_history:
            return None
        normalized_current = _normalize_regime_name(current_regime)
        last_alert = self._alert_history[-1]
        # 若最近一笔 trade 是盈利 **且** regime 与警报一致, 警报消解
        # (MEDIUM #5: 跨 regime 盈利不消解, 因为本警报是针对特定 regime 的连续亏损)
        if self._recent_trades:
            last_pnl, last_regime = self._recent_trades[-1]
            if (last_pnl > self.loss_pnl_threshold
                    and last_regime == last_alert.regime):
                return None  # 同 regime 盈利, 警报消解
        # 若警报 regime 与当前 regime 匹配, 返回活跃警报
        if last_alert.regime == normalized_current:
            return last_alert
        return None

    def get_alert_history(self) -> List[Dict[str, Any]]:
        """获取历史警报 (供 round_results.warnings 写入)"""
        return [a.to_dict() for a in self._alert_history]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        n_losses = sum(1 for p, _ in self._recent_trades if p <= self.loss_pnl_threshold)
        n_wins = len(self._recent_trades) - n_losses
        return {
            "total_trades_observed": self._trade_counter,
            "window_size": len(self._recent_trades),
            "window_losses": n_losses,
            "window_wins": n_wins,
            "alert_count": len(self._alert_history),
            "last_alert": self._alert_history[-1].to_dict() if self._alert_history else None,
        }

    def _detect_consec_loss_alert(self, regime: str) -> Optional[FailureAlert]:
        """检测同 regime 下的连续亏损 (核心检测逻辑)"""
        # 从窗口末尾向前扫描, 找出最近连续亏损段
        # 仅统计同 regime 的 trades (跨 regime 不算连续)
        consec_losses: List[float] = []
        for pnl, t_regime in reversed(self._recent_trades):
            if t_regime != regime:
                break
            if pnl <= self.loss_pnl_threshold:
                consec_losses.append(pnl)
            else:
                break  # 遇到盈利, 中断

        if len(consec_losses) < self.consec_loss_threshold:
            return None

        # v598 Phase J MEDIUM #6 修复: 用 cooldown 窗口替代单一 idx 防护
        # 原 bug: _last_alert_trade_idx == self._trade_counter 仅阻止同一 trade 重复触发,
        #         下笔亏损 (trade_counter+1) 仍会再次触发警报, 因为窗口仍包含连续亏损段.
        # 修复: 触发后 alert_cooldown 笔 trade 内不再触发新警报 (给策略调整留观察期)
        # 例: trade_idx=10 触发警报, cooldown=5 → trade_idx 11..15 不再触发, 16 起恢复检测
        if (self._trade_counter - self._last_alert_trade_idx) <= self.alert_cooldown:
            return None  # 冷却期内, 不重复触发

        failure_mode = _REGIME_TO_FAILURE_MODE.get(regime)
        if failure_mode is None:
            return None  # 未知 regime, 无匹配失败模式 (不消耗 cooldown)

        # v598 Phase J MINOR #7 修复 (L7 审查发现):
        #   原 bug: _last_alert_trade_idx 赋值在 failure_mode None 检查之前,
        #           导致 trend_strong/uptrend 等无 failure_mode 的 regime 连续亏损
        #           仍会消耗 cooldown, 阻塞后续 valid 警报.
        #   修复: 将赋值移到 failure_mode None 检查之后, 确保只有真正触发警报时才消耗 cooldown.
        self._last_alert_trade_idx = self._trade_counter

        recommended = self._REGIME_RECOMMENDED_PARADIGM.get(
            regime, StrategyParadigm.REDUCED_POSITION,
        )
        root_cause_hint = self._REGIME_ROOT_CAUSE_HINT.get(regime, "未知根因")

        consec_loss_pnl = float(sum(consec_losses))
        avg_loss = consec_loss_pnl / len(consec_losses) if consec_losses else 0.0

        return FailureAlert(
            detected_at_trade_idx=self._trade_counter,
            regime=regime,
            failure_mode=failure_mode,
            consec_loss_count=len(consec_losses),
            consec_loss_pnl=consec_loss_pnl,
            avg_loss=avg_loss,
            recommended_paradigm=recommended,
            root_cause_hint=root_cause_hint,
        )


# ============================================================================
# 5. 主类: RegimeFailureUnderstanding
# ============================================================================

class RegimeFailureUnderstanding:
    """市场状态失败理解层

    在 RegimeDetectionSystem 之上构建认知层, 实现:
      1. 根因分析: 当前 regime 下为何会失败(基于知识库+动态学习)
      2. 策略切换建议: 从失效范式切换到有效范式
      3. 仓位连续调整: 基于失败概率的连续值 [0.3, 1.5]
      4. 指标失效预警: 哪些指标在当前 regime 下会失效
      5. 反事实推理: 若切换策略, 预期胜率提升多少

    输入:
      - regime_data: 来自 RegimeDetectionSystem.update()
      - failure_history: 失败交易历史(来自 sandbox_engine.trade_history)
      - current_indicators: 当前指标状态

    输出:
      - FailureUnderstandingReport
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, RegimeFailurePattern]] = None,
        failure_history_window: int = 100,
        min_failures_for_learning: int = 5,
        dynamic_learning_rate: float = 0.1,
        confidence_threshold: float = 0.6,
    ):
        """
        Args:
            knowledge_base: 失败模式知识库, 默认使用 REGIME_FAILURE_KNOWLEDGE_BASE
            failure_history_window: 失败历史窗口大小
            min_failures_for_learning: 触发动态学习的最小失败数
            dynamic_learning_rate: 动态学习率(0-1, 越大越依赖动态)
            confidence_threshold: 失效预警的置信度阈值
        """
        self.knowledge_base = knowledge_base or REGIME_FAILURE_KNOWLEDGE_BASE
        self.failure_history_window = failure_history_window
        self.min_failures_for_learning = min_failures_for_learning
        self.dynamic_learning_rate = dynamic_learning_rate
        self.confidence_threshold = confidence_threshold

        # 失败交易历史(按 regime 分组)
        self._failure_history: deque = deque(maxlen=failure_history_window)
        # 动态失效概率(覆盖知识库的静态值)
        self._dynamic_failure_probs: Dict[str, Dict[str, float]] = {}

        logger.info(
            f"RegimeFailureUnderstanding 初始化: "
            f"knowledge_base={len(self.knowledge_base)} patterns, "
            f"window={failure_history_window}, "
            f"min_failures={min_failures_for_learning}, "
            f"lr={dynamic_learning_rate}"
        )

    # ----------------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------------

    def understand(
        self,
        regime_data: Dict[str, Any],
        current_indicators: Dict[str, float],
        current_strategy_paradigm: StrategyParadigm,
    ) -> FailureUnderstandingReport:
        """主入口: 理解当前 regime 下的失败模式并给出建议

        Args:
            regime_data: RegimeDetectionSystem.update() 返回的状态字典
            current_indicators: 当前各指标值
            current_strategy_paradigm: 当前使用的策略范式

        Returns:
            FailureUnderstandingReport
        """
        market_regime = regime_data.get("market_regime", "sideways")
        vol_regime = regime_data.get("volatility_state", {}).get("volatility_regime", "NORMAL")
        is_transitioning = regime_data.get("is_transitioning", False)

        # Step1: 检测趋势末端(特殊场景)
        effective_regime = self._detect_uptrend_end(market_regime, current_indicators)

        # Step2: 查找失败模式
        pattern = self.knowledge_base.get(effective_regime)
        if pattern is None:
            return self._default_report(regime_data, current_strategy_paradigm)

        # Step3: 动态学习(如果有足够失败历史)
        dynamic_probs = self._get_dynamic_failure_probs(effective_regime)
        merged_probs = self._merge_static_dynamic(
            pattern.indicator_failure_predictors, dynamic_probs
        )

        # Step4: 根因分析
        root_cause = self._analyze_root_cause(
            pattern, current_indicators, merged_probs, effective_regime
        )

        # Step5: 策略切换建议
        strategy_switch = self._recommend_strategy_switch(
            pattern, current_strategy_paradigm, effective_regime, vol_regime
        )

        # Step6: 仓位连续调整
        position_coef = self._compute_continuous_position(
            pattern, regime_data, dynamic_probs, is_transitioning
        )

        # Step7: 指标失效预警
        failure_warnings = self._generate_failure_warnings(
            merged_probs, current_indicators, confidence_threshold=self.confidence_threshold
        )

        # Step8: 反事实推理
        counterfactual = self._counterfactual_reasoning(
            pattern, current_strategy_paradigm, strategy_switch.recommended_paradigm
        )

        return FailureUnderstandingReport(
            effective_regime=effective_regime,
            detected_failure_mode=pattern.failure_mode,
            root_cause=root_cause,
            current_paradigm=current_strategy_paradigm,
            recommended_paradigm=strategy_switch.recommended_paradigm,
            strategy_switch_confidence=strategy_switch.confidence,
            indicator_failure_probs=merged_probs,
            failure_warnings=failure_warnings,
            recommended_sl_multiplier=pattern.recommended_sl_multiplier,
            recommended_tp_multiplier=pattern.recommended_tp_multiplier,
            recommended_position_coef=position_coef,
            counterfactual_win_rate_delta=counterfactual,
            regime_confidence=regime_data.get("market_confidence", 0.0),
            is_transitioning=is_transitioning,
        )

    def record_failure(self, record: FailureTradeRecord) -> None:
        """记录失败交易, 触发动态学习"""
        self._failure_history.append(record)
        if len(self._failure_history) >= self.min_failures_for_learning:
            self._update_dynamic_probs()

    def get_failure_stats(self) -> Dict[str, Any]:
        """获取失败统计"""
        by_regime: Dict[str, int] = {}
        for record in self._failure_history:
            by_regime[record.regime] = by_regime.get(record.regime, 0) + 1
        return {
            "total_failures": len(self._failure_history),
            "by_regime": by_regime,
            "dynamic_failure_probs": self._dynamic_failure_probs,
        }

    # ====================================================================
    # v596 C3: 按 regime 分组失败分析 + 自动写入 v595 知识库
    # ====================================================================

    def analyze_failures_by_regime(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """v596 C3: 按 regime 分组统计失败模式

        将交易记录按标准 6 状态分组，计算每个 regime 下的:
          - 交易数/胜率/总PnL/平均PnL
          - 是否为失败 regime (win_rate<0.4 或 total_pnl<0)
          - 匹配的已知失败模式列表

        Args:
            trades: 交易记录列表，每条需含 regime/pnl_pct 字段

        Returns:
            {regime_normalized: {n_trades, win_rate, total_pnl_pct, avg_pnl_pct,
                                  is_failure, failure_patterns}}
        """
        from _v596_regime_namespace import normalize_regime
        from collections import defaultdict

        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for t in trades:
            regime = normalize_regime(t.get("regime", "unknown")).value
            grouped[regime].append(t)

        results: Dict[str, Dict[str, Any]] = {}
        for regime, regime_trades in grouped.items():
            n = len(regime_trades)
            wins = sum(1 for t in regime_trades if float(t.get("pnl_pct", 0)) > 0)
            total_pnl = sum(float(t.get("pnl_pct", 0)) for t in regime_trades)
            avg_pnl = total_pnl / n if n > 0 else 0.0
            win_rate = wins / n if n > 0 else 0.0

            results[regime] = {
                "n_trades": n,
                "win_rate": round(win_rate, 4),
                "total_pnl_pct": round(total_pnl, 4),
                "avg_pnl_pct": round(avg_pnl, 4),
                "is_failure": win_rate < 0.4 or total_pnl < 0,
                "failure_patterns": self._detect_failure_patterns(regime, regime_trades),
            }
        return results

    def _detect_failure_patterns(self, regime: str, trades: List[Dict[str, Any]]) -> List[str]:
        """v596 C3: 检测 4 种已知 regime 失败模式的匹配情况

        已知失败模式 (来自 RegimeFailurePattern 知识库):
          1. ranging_failure: 震荡市假突破 (突破后回撤大)
          2. volatile_failure: 高波动止损频繁 (hold_bars 短 + 亏损)
          3. downtrend_failure: 逆势做多 (direction=1 + regime=trending_down)
          4. uptrend_end_failure: 趋势末端反转 (regime=trending_up + 大亏损)

        Args:
            regime: 标准化 regime 名 (trending_up/down, ranging, volatile, crisis, accumulation)
            trades: 该 regime 下的交易记录

        Returns:
            匹配的失败模式列表 (空列表表示无匹配)
        """
        patterns: List[str] = []

        if not trades:
            return patterns

        # 统计指标
        n = len(trades)
        losses = [t for t in trades if float(t.get("pnl_pct", 0)) < 0]
        avg_loss = (
            sum(float(t.get("pnl_pct", 0)) for t in losses) / len(losses)
            if losses else 0.0
        )
        avg_hold = sum(int(t.get("hold_bars", 0)) for t in trades) / n if n > 0 else 0

        # 模式 1: ranging 假突破 — 震荡市中突破后回撤
        if regime == "ranging" and len(losses) > n * 0.5:
            patterns.append("ranging_false_breakout")

        # 模式 2: volatile 频繁止损 — 高波动下 hold_bars 短 + 亏损多
        if regime == "volatile" and avg_hold < 6 and len(losses) > n * 0.5:
            patterns.append("volatile_frequent_stoploss")

        # 模式 3: downtrend 逆势做多 — 下降趋势中 direction=1 的交易亏损
        if regime == "trending_down":
            long_losses = [
                t for t in losses
                if int(t.get("direction", 0)) == 1
            ]
            if len(long_losses) > 2:
                patterns.append("downtrend_counter_trend_long")

        # 模式 4: uptrend_end 趋势末端 — 上升趋势中单笔大亏损
        if regime == "trending_up" and avg_loss < -0.02:
            patterns.append("uptrend_end_reversal")

        # 模式 5: crisis 全局失效 — 危机 regime 中任何交易都是失败
        if regime == "crisis" and len(losses) > 0:
            patterns.append("crisis_all_failure")

        return patterns

    def write_failures_to_kb(self, failures: Dict[str, Dict[str, Any]]) -> List[str]:
        """v596 C3: 自动将 regime 失败模式写入 v595 全局知识库

        仅对 is_failure=True 的 regime 写入 ERR 条目，
        调用 _v595_global_knowledge_base.add_regime_failure_err 接口。

        Args:
            failures: analyze_failures_by_regime 的返回值

        Returns:
            写入的 ERR-ID 列表
        """
        written_errs: List[str] = []
        try:
            from _v595_global_knowledge_base import add_regime_failure_err
        except ImportError:
            logger.warning("v596 C3: 无法导入 add_regime_failure_err, 跳过 KB 写入")
            return written_errs

        for regime, stats in failures.items():
            if not stats.get("is_failure", False):
                continue
            err_id = f"ERR-REGIME-{regime.upper()}-{stats['n_trades']}T"
            try:
                add_regime_failure_err(
                    err_id=err_id,
                    regime=regime,
                    win_rate=stats["win_rate"],
                    total_pnl=stats["total_pnl_pct"],
                    patterns=stats.get("failure_patterns", []),
                )
                written_errs.append(err_id)
                logger.info("v596 C3: 写入 regime 失败 ERR=%s (regime=%s, win_rate=%.2f)",
                            err_id, regime, stats["win_rate"])
            except Exception as e:
                logger.warning("v596 C3: 写入 ERR=%s 失败: %s", err_id, e)

        return written_errs

    # ----------------------------------------------------------------
    # 私有方法(核心逻辑)
    # ----------------------------------------------------------------

    def _detect_uptrend_end(
        self, market_regime: str, indicators: Dict[str, float]
    ) -> str:
        """检测趋势末端(bull + 动量衰减 → uptrend_end) + regime 别名归一化

        v447 修复: regime 别名映射
          原始问题: RegimeDetectionSystem 输出 'sideways'/'bull'/'bear'/'volatile',
                   但知识库 key 是 'ranging'/'uptrend'/'downtrend'/'volatile',
                   导致 'sideways' 场景永远匹配不到 'ranging' 失败模式.
          修复: 添加别名归一化层, 让两种命名互通.
        """
        # v447: regime 别名归一化 (RegimeDetectionSystem 命名 ↔ 知识库命名)
        REGIME_ALIASES = {
            "sideways": "ranging",       # HMM 震荡 ↔ 知识库 ranging
            "range": "ranging",
            "consolidation": "ranging",
            "bull": "uptrend",           # HMM 牛市 ↔ 知识库 uptrend
            "uptrend": "uptrend",
            "bear": "downtrend",         # HMM 熊市 ↔ 知识库 downtrend
            "downtrend": "downtrend",
            "volatile": "volatile",      # 高波动 (同名)
            "ranging": "ranging",
            "uptrend_end": "uptrend_end",
        }
        market_regime = REGIME_ALIASES.get(market_regime, market_regime)

        # 检测 uptrend_end (bull + 动量衰减)
        if market_regime == "uptrend":
            # 检测 MACD 柱缩短(动量衰减)
            macd_hist = indicators.get("macd_histogram", indicators.get("macd_hist", 0))
            macd_hist_prev = indicators.get("macd_histogram_prev", 0)
            if macd_hist > 0 and macd_hist < macd_hist_prev * 0.7:
                logger.debug("检测到 uptrend 末端: MACD 柱缩短, 动量衰减")
                return "uptrend_end"
        return market_regime

    def _get_dynamic_failure_probs(self, regime: str) -> Dict[str, float]:
        """获取动态学习的失效概率"""
        return self._dynamic_failure_probs.get(regime, {})

    def _merge_static_dynamic(
        self, static: Dict[str, float], dynamic: Dict[str, float]
    ) -> Dict[str, float]:
        """融合静态知识库 + 动态学习(指数加权)"""
        merged = {}
        all_keys = set(static.keys()) | set(dynamic.keys())
        for key in all_keys:
            s = static.get(key, 0.5)
            d = dynamic.get(key, s)
            # 动态权重 = learning_rate
            merged[key] = (1 - self.dynamic_learning_rate) * s + self.dynamic_learning_rate * d
        return merged

    def _analyze_root_cause(
        self,
        pattern: RegimeFailurePattern,
        indicators: Dict[str, float],
        failure_probs: Dict[str, float],
        regime: str,
    ) -> str:
        """根因分析: 构建因果链"""
        # 找出当前激活的指标中失效概率最高的
        active_high_failure = []
        for ind_name, ind_value in indicators.items():
            if ind_name in failure_probs and failure_probs[ind_name] > 0.6:
                active_high_failure.append((ind_name, failure_probs[ind_name]))

        active_high_failure.sort(key=lambda x: -x[1])
        top_failures = active_high_failure[:3]

        cause_chain = []
        cause_chain.append(f"当前 regime={regime}")
        cause_chain.append(f"已知根因: {pattern.root_cause}")
        if top_failures:
            ind_str = ", ".join([f"{n}(失效概率={p:.0%})" for n, p in top_failures])
            cause_chain.append(f"当前激活的高失效指标: {ind_str}")
        cause_chain.append(
            f"建议范式: {pattern.effective_paradigms[0].value} "
            f"(替代失效的 {pattern.failed_paradigms[0].value})"
        )
        return " → ".join(cause_chain)

    def _recommend_strategy_switch(
        self,
        pattern: RegimeFailurePattern,
        current: StrategyParadigm,
        regime: str,
        vol_regime: str,
    ) -> StrategySwitchRecommendation:
        """策略切换建议"""
        if current in pattern.effective_paradigms:
            # 当前范式在该 regime 下有效, 无需切换
            return StrategySwitchRecommendation(
                recommended_paradigm=current,
                confidence=0.9,
                reason=f"当前范式 {current.value} 在 {regime} 下有效, 无需切换",
            )

        # 当前范式失效, 建议切换
        if pattern.effective_paradigms:
            recommended = pattern.effective_paradigms[0]
            # 高波动下进一步降级
            if vol_regime == "HIGH" and recommended != StrategyParadigm.REDUCED_POSITION:
                recommended = StrategyParadigm.REDUCED_POSITION
            return StrategySwitchRecommendation(
                recommended_paradigm=recommended,
                confidence=0.75,
                reason=(
                    f"当前范式 {current.value} 在 {regime} 下失效, "
                    f"切换到 {recommended.value}"
                ),
            )

        return StrategySwitchRecommendation(
            recommended_paradigm=StrategyParadigm.REDUCED_POSITION,
            confidence=0.5,
            reason="无有效范式, 建议减仓观察",
        )

    def _compute_continuous_position(
        self,
        pattern: RegimeFailurePattern,
        regime_data: Dict[str, Any],
        dynamic_probs: Dict[str, float],
        is_transitioning: bool,
    ) -> float:
        """连续仓位调整(基于失败概率+波动率+转换期)"""
        # 基础值来自知识库
        base = pattern.recommended_position_coef

        # 根据当前激活指标的平均失效概率调整
        avg_failure = float(np.mean(list(dynamic_probs.values()))) if dynamic_probs else 0.5
        failure_penalty = 1.0 - 0.3 * avg_failure   # 失效概率越高, 仓位越低

        # 波动率调整
        vol_state = regime_data.get("volatility_state", {})
        vol_adj = vol_state.get("position_adjustment", 1.0)

        # 转换期额外降仓
        transition_penalty = 0.7 if is_transitioning else 1.0

        # 综合调整
        final = base * failure_penalty * vol_adj * transition_penalty

        # 钳制到合理范围
        return max(0.3, min(1.5, final))

    def _generate_failure_warnings(
        self,
        failure_probs: Dict[str, float],
        indicators: Dict[str, float],
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:
        """生成指标失效预警"""
        warnings = []
        for ind_name, prob in failure_probs.items():
            if prob > confidence_threshold and ind_name in indicators:
                warnings.append({
                    "indicator": ind_name,
                    "failure_probability": prob,
                    "current_value": indicators[ind_name],
                    "warning_level": "HIGH" if prob > 0.8 else "MEDIUM",
                })
        return sorted(warnings, key=lambda x: -x["failure_probability"])

    def _counterfactual_reasoning(
        self,
        pattern: RegimeFailurePattern,
        current: StrategyParadigm,
        recommended: StrategyParadigm,
    ) -> float:
        """反事实推理: 切换策略后预期胜率提升

        Returns:
            预期胜率提升幅度(如 +0.15 表示提升 15%)
        """
        if current == recommended:
            return 0.0
        # 基于知识库的静态估计
        # failed_paradigm 在该 regime 下胜率通常 < 20%
        # effective_paradigm 在该 regime 下胜率通常 > 50%
        # 预期提升 ≈ 0.30~0.40
        base_delta = 0.30
        # 根据失败概率调整
        avg_failure = float(np.mean(list(pattern.indicator_failure_predictors.values())))
        return base_delta * avg_failure

    def _update_dynamic_probs(self) -> None:
        """从失败历史中动态更新失效概率"""
        by_regime: Dict[str, List[FailureTradeRecord]] = {}
        for record in self._failure_history:
            by_regime.setdefault(record.regime, []).append(record)

        for regime, records in by_regime.items():
            if len(records) < self.min_failures_for_learning:
                continue
            # 统计每个指标在失败交易中的出现率
            indicator_failure_count: Dict[str, int] = {}
            for r in records:
                for ind in r.indicators_used:
                    indicator_failure_count[ind] = indicator_failure_count.get(ind, 0) + 1
            # 转换为概率
            total = len(records)
            self._dynamic_failure_probs[regime] = {
                ind: count / total for ind, count in indicator_failure_count.items()
            }
            logger.info(
                f"动态学习: regime={regime}, 失败记录={total} 条, "
                f"更新 {len(indicator_failure_count)} 个指标失效概率"
            )

    def _default_report(
        self, regime_data: Dict[str, Any], current: StrategyParadigm
    ) -> FailureUnderstandingReport:
        """默认报告(无知识库匹配时)"""
        return FailureUnderstandingReport(
            effective_regime=regime_data.get("market_regime", "unknown"),
            detected_failure_mode=None,
            root_cause="无匹配的失败模式知识库条目",
            current_paradigm=current,
            recommended_paradigm=current,
            strategy_switch_confidence=0.0,
            indicator_failure_probs={},
            failure_warnings=[],
            recommended_sl_multiplier=1.0,
            recommended_tp_multiplier=1.0,
            recommended_position_coef=1.0,
            counterfactual_win_rate_delta=0.0,
            regime_confidence=regime_data.get("market_confidence", 0.0),
            is_transitioning=regime_data.get("is_transitioning", False),
        )


# ============================================================================
# 6. 工具函数: worldview → StrategyParadigm 转换
# ============================================================================

_WORLDVIEW_TO_PARADIGM_MAP: Dict[str, StrategyParadigm] = {
    # 趋势跟随
    "trend_following": StrategyParadigm.TREND_FOLLOWING,
    "momentum": StrategyParadigm.TREND_FOLLOWING,
    "trend": StrategyParadigm.TREND_FOLLOWING,
    # 均值回归
    "mean_reversion": StrategyParadigm.MEAN_REVERSION,
    "mean_reversion_ml": StrategyParadigm.MEAN_REVERSION,
    "stat_arb_pairs": StrategyParadigm.MEAN_REVERSION,
    "stat_arbitrage": StrategyParadigm.MEAN_REVERSION,
    # 剥头皮
    "scalping": StrategyParadigm.SCALPING,
    "high_frequency": StrategyParadigm.SCALPING,
    # 减仓防守
    "safe_margin": StrategyParadigm.REDUCED_POSITION,
    "risk_parity": StrategyParadigm.REDUCED_POSITION,
    "risk_parity_signal": StrategyParadigm.REDUCED_POSITION,
    # 顺势做空
    "value_investing": StrategyParadigm.COUNTER_TREND_SHORT,
    # 突破
    "breakout": StrategyParadigm.BREAKOUT,
    "chanlun_priceaction": StrategyParadigm.BREAKOUT,
}


def worldview_to_paradigm(worldview: str) -> StrategyParadigm:
    """将 worldview 名称转换为 StrategyParadigm

    Args:
        worldview: Agent 的世界观名称(如 "trend_following", "mean_reversion")

    Returns:
        对应的 StrategyParadigm 枚举值, 默认为 TREND_FOLLOWING
    """
    return _WORLDVIEW_TO_PARADIGM_MAP.get(worldview.lower(), StrategyParadigm.TREND_FOLLOWING)


# ============================================================================
# 7. 单元测试入口
# ============================================================================

def _self_test() -> bool:
    """自检: 验证核心功能可用"""
    try:
        understander = RegimeFailureUnderstanding()

        # 测试1: ranging regime 理解
        regime_data = {
            "market_regime": "ranging",
            "market_confidence": 0.8,
            "is_transitioning": False,
            "volatility_state": {"volatility_regime": "NORMAL", "position_adjustment": 1.0},
        }
        indicators = {"macd_cross": 1.0, "rsi_divergence": 0.3, "bollinger_squeeze": 0.5}
        report = understander.understand(
            regime_data=regime_data,
            current_indicators=indicators,
            current_strategy_paradigm=StrategyParadigm.TREND_FOLLOWING,
        )

        assert report.effective_regime == "ranging"
        assert report.detected_failure_mode == FailureMode.RANGING_SIGNAL_NOISE
        assert report.recommended_paradigm == StrategyParadigm.MEAN_REVERSION
        assert report.should_switch_strategy() is True
        assert 0.3 <= report.recommended_position_coef <= 1.5

        # 测试2: volatile regime 理解
        regime_data_volatile = {
            "market_regime": "volatile",
            "market_confidence": 0.6,
            "is_transitioning": True,
            "volatility_state": {"volatility_regime": "HIGH", "position_adjustment": 0.6},
        }
        report_volatile = understander.understand(
            regime_data=regime_data_volatile,
            current_indicators={"macd_cross": 1.0, "volume_anomaly": 0.5},
            current_strategy_paradigm=StrategyParadigm.SCALPING,
        )
        assert report_volatile.recommended_paradigm == StrategyParadigm.REDUCED_POSITION
        assert report_volatile.recommended_position_coef < 0.6  # 高波动+转换期应更低

        # 测试3: worldview 转换
        assert worldview_to_paradigm("trend_following") == StrategyParadigm.TREND_FOLLOWING
        assert worldview_to_paradigm("mean_reversion") == StrategyParadigm.MEAN_REVERSION

        # 测试4: 失败记录
        for i in range(10):
            understander.record_failure(FailureTradeRecord(
                timestamp=float(i),
                regime="ranging",
                strategy_paradigm=StrategyParadigm.TREND_FOLLOWING,
                indicators_used=["macd_cross", "ema_breakout"],
                pnl=-1.0,
                failure_reason="stop_loss",
                indicators_state={"macd_cross": 1.0, "ema_breakout": 1.0},
            ))
        stats = understander.get_failure_stats()
        assert stats["total_failures"] == 10
        assert "ranging" in stats["by_regime"]

        logger.info("RegimeFailureUnderstanding 自检全部通过 ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    _self_test()
