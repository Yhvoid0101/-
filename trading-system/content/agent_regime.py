"""Agent regime mixin (Phase 7.14c extracted from agent_decision_engine.py).

Contains 3 regime/market-state analysis methods used by decide().
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("hermes.agent_decision_engine")

# Phase 7.14c: Re-define _RFU_AVAILABLE here (same pattern as Phase 7.10
# _STAGED_ADMISSION_AVAILABLE). The original try/except is in
# agent_decision_engine.py but this Mixin needs its own copy because the
# _apply_regime_failure_understanding method references _RFU_AVAILABLE.
try:
    from .regime_failure_understanding import (
        RegimeFailureUnderstanding,
        StrategyParadigm,
        FailureMode,
    )
    _RFU_AVAILABLE = True
except ImportError as _rfu_err:
    _RFU_AVAILABLE = False
    RegimeFailureUnderstanding = None  # type: ignore
    StrategyParadigm = None  # type: ignore
    FailureMode = None  # type: ignore


class AgentRegimeMixin:
    """Mixin providing regime/market-state analysis methods for AgentDecisionEngine.

    Extracted in Phase 7.14c to reduce agent_decision_engine.py size.
    All methods use duck-typing (self attribute access) - zero circular deps.
    """

    def _apply_regime_failure_understanding(
        self,
        regime_data: Optional[Dict[str, Any]],
        indicators: Dict[str, float],
        start: float,
    ) -> Optional[Dict[str, Any]]:
        """v447 认知层: 应用 Regime 失败理解 (从"过滤"到"理解后切换")

        用户明确要求:
          "不只是过滤市场状态, 而是理解为什么在这些市场状态失败"
          "不只是强制 confluence, 而是理解各指标的底层逻辑"

        核心逻辑:
          1. 调用 RegimeFailureUnderstanding.understand() 获取 failure_report
          2. 如果失败模式被识别:
             - 切换策略范式 (如 trend_following → mean_reversion)
             - 调整 SL/TP/仓位系数 (基于失败模式知识库)
             - 标记 regime_adjustment 供后续信号融合使用
          3. 仅在"完全无有效策略"时返回 None (弃权)

        Args:
            regime_data: RegimeDetectionSystem 返回的状态字典 (可能为 None)
            indicators: 当前技术指标字典
            start: decide() 开始时间戳 (用于耗时记录)

        Returns:
            调整字典 {
                "signal_score_multiplier": float,    # 信号得分乘数
                "position_size_multiplier": float,   # 仓位系数
                "sl_multiplier": float,              # 止损乘数
                "tp_multiplier": float,              # 止盈乘数
                "strategy_paradigm": str,            # 建议的策略范式
                "failure_mode": str,                 # 失败模式 (如有)
                "root_cause": str,                   # 根因 (如有)
            }
            或 None 表示完全弃权 (无有效策略)
        """
        # 默认调整: 不变 (向后兼容)
        default_adjustment = {
            "signal_score_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "sl_multiplier": 1.0,
            "tp_multiplier": 1.0,
            "strategy_paradigm": "default",
            "failure_mode": "none",
            "root_cause": "",
        }

        if not regime_data:
            return default_adjustment

        # RegimeFailureUnderstanding 初始化 (v598 Phase 5: 改用顶层导入, 修复裸路径脆弱点)
        # 原缺陷: sys.path.insert + 裸 from regime_failure_understanding import + logger.debug 静默降级
        # 修复: 顶层已 try/except 相对导入 (_RFU_AVAILABLE 标志), 这里只做实例化
        if self._regime_failure_understander is None:
            if not _RFU_AVAILABLE:
                # 顶层导入已失败, 降级 (warning 已在顶层记录)
                self._regime_failure_understander = False
            else:
                try:
                    self._regime_failure_understander = RegimeFailureUnderstanding()
                    # 从基因推导当前策略范式 (用顶层导入的枚举)
                    self._rfu_strategy_paradigm_enum = StrategyParadigm
                    self._rfu_failure_mode_enum = FailureMode
                except Exception as e:
                    logger.warning("RegimeFailureUnderstanding 实例化失败 (6大短板#6 降级): %s", e)
                    self._regime_failure_understander = False

        if self._regime_failure_understander is False or self._regime_failure_understander is None:
            return default_adjustment

        # 从基因推导当前策略范式
        worldview = getattr(self.gene, "worldview_primary", "").lower()
        if any(k in worldview for k in ("trend", "momentum", "breakout")):
            current_paradigm = self._rfu_strategy_paradigm_enum.TREND_FOLLOWING
        elif any(k in worldview for k in ("mean_reversion", "reversion", "stat_arb")):
            current_paradigm = self._rfu_strategy_paradigm_enum.MEAN_REVERSION
        elif any(k in worldview for k in ("scalping", "market_making")):
            current_paradigm = self._rfu_strategy_paradigm_enum.SCALPING
        else:
            current_paradigm = self._rfu_strategy_paradigm_enum.TREND_FOLLOWING

        # v598 Phase J MEDIUM #2 修复: 在 try 外初始化, 保证后续 signal_score_mult 衰减可见
        _phase_j_active_alert: Optional[Dict[str, Any]] = None

        try:
            # 调用认知层 understand() 方法 (返回 FailureUnderstandingReport 对象)
            report_obj = self._regime_failure_understander.understand(
                regime_data=regime_data,
                current_indicators=indicators,
                current_strategy_paradigm=current_paradigm,
            )
            # 转换为字典 (字段名: failure_mode/root_cause/recommended_paradigm/
            #   sl_multiplier/tp_multiplier/position_coef/strategy_switch_confidence/
            #   indicator_failure_probs/failure_warnings 等)
            failure_report = report_obj.to_dict() if hasattr(report_obj, "to_dict") else {}
            self._last_failure_report = failure_report

            # v598 Phase J: 查询 FailureModeMatcher 是否有活跃警报 (自动联动闭环)
            # 闭环: loop_end notify_trade_closed → matcher 累积 → decide() detect_active_failure_alert
            # ERR-110 边界: 警报仅作为信号增强, 不直接 block 交易
            try:
                _current_regime = regime_data.get("market_regime", "")
                _active_alert = self.detect_active_failure_alert(_current_regime)
                if _active_alert is not None:
                    failure_report["failure_alert"] = _active_alert
                    # v598 Phase J MEDIUM #2 修复: 不再写入 dead code 字段 signal_mult_hint
                    # 10% 信号衰减改为在下方 signal_score_mult 计算后实际叠加 (L2078+)
                    _phase_j_active_alert = _active_alert  # 传递给后续 adjustment 构造
                    logger.info(
                        "v598 Phase J: agent=%s 活跃警报 regime=%s failure_mode=%s "
                        "consec_loss=%d, recommended=%s",
                        getattr(self.gene, 'agent_id', '?'),
                        _active_alert.get("regime", "?"),
                        _active_alert.get("failure_mode", "?"),
                        int(_active_alert.get("consec_loss_count", 0)),
                        _active_alert.get("recommended_paradigm", "?"),
                    )
            except Exception as _alert_err:
                logger.debug("FailureModeMatcher detect_active_failure_alert 异常: %s", _alert_err)
        except Exception as e:
            logger.debug("RegimeFailureUnderstanding.understand() 异常: %s", e)
            return default_adjustment

        # 如果没有失败模式被识别, 返回默认调整
        if not failure_report or not failure_report.get("failure_mode"):
            return default_adjustment

        failure_mode_str = failure_report.get("failure_mode", "")
        root_cause = failure_report.get("root_cause", "")
        recommended_paradigm = failure_report.get("recommended_paradigm", "")
        sl_mult = float(failure_report.get("sl_multiplier", 1.0))
        tp_mult = float(failure_report.get("tp_multiplier", 1.0))
        pos_coef = float(failure_report.get("position_coef", 1.0))
        switch_conf = float(failure_report.get("strategy_switch_confidence", 0.0))

        # 完全弃权条件:
        #   1. 推荐范式为 REDUCED_POSITION (减仓防守)
        #   2. 且切换置信度 > 0.7 (高置信度判断无有效策略)
        # 设计哲学: 仅在认知层高置信度判断"完全无有效策略"时才弃权,
        #          否则切换到推荐范式继续交易 (理解后切换, 而非硬过滤)
        if recommended_paradigm == "reduced_position" and switch_conf > 0.7:
            return None

        # 信号得分乘数: 失败模式下减弱原策略信号
        # 来源: indicator_failure_probs 的均值 (失败概率越高, 信号越弱)
        indicator_probs = failure_report.get("indicator_failure_probs", {})
        if indicator_probs:
            avg_failure_prob = float(np.mean(list(indicator_probs.values())))
            signal_score_mult = 1.0 - avg_failure_prob * 0.5  # [0.5, 1.0]
        else:
            signal_score_mult = 0.7  # 默认减弱 30%

        # v598 Phase J MEDIUM #2 修复: 活跃警报额外叠加 10% 信号衰减 (实际生效, 替代原 dead code)
        # 原 bug: signal_mult_hint 字段无读取方, 声称的 10% 衰减从未生效
        # 修复: 直接修改 signal_score_mult, 该值通过 adjustment["signal_score_multiplier"]
        #       在 L882-883 被 signal_score *= ... 真实消费
        # ERR-110 边界: 仍属"信号增强"范畴 (受 clip_range 约束), 不修改入场/出场 threshold
        if _phase_j_active_alert is not None:
            _alert_consec = int(_phase_j_active_alert.get("consec_loss_count", 0))
            if _alert_consec >= 3:
                signal_score_mult *= 0.9  # 活跃警报额外衰减 10%

        adjustment = {
            "signal_score_multiplier": signal_score_mult,
            "position_size_multiplier": pos_coef,
            "sl_multiplier": sl_mult,
            "tp_multiplier": tp_mult,
            "strategy_paradigm": str(recommended_paradigm),
            "failure_mode": str(failure_mode_str),
            "root_cause": str(root_cause),
        }

        # 每1000次输出一次认知层诊断
        if not hasattr(self, '_rfu_diag_counter'):
            self._rfu_diag_counter = 0
        self._rfu_diag_counter += 1
        if self._rfu_diag_counter % 1000 == 1:
            logger.warning(
                "v447 RFU DIAG: agent=%s regime=%s failure=%s paradigm=%s→%s "
                "sl×%.2f tp×%.2f pos×%.2f signal×%.2f conf=%.2f",
                getattr(self.gene, 'agent_id', '?'),
                regime_data.get("market_regime", "?"),
                failure_mode_str,
                current_paradigm.value,
                recommended_paradigm,
                sl_mult, tp_mult, pos_coef, signal_score_mult, switch_conf,
            )

        return adjustment

    def _compute_regime_aware_coef(
        self,
        regime_data: Dict[str, Any],
        default: float = 0.15,
    ) -> float:
        """计算 regime-aware 策略信号增强系数（Phase 6B）

        根据 market_regime + risk_level + is_transitioning 动态调整系数：
          - 趋势市场（bull/bear）+ 低风险：系数↑（趋势策略更有效）
          - 震荡市场（sideways）：系数↓（趋势策略效果差）
          - 风险等级越高：系数越低（避免假突破）
          - 状态转换期：再乘 0.5（不确定性高）

        来源：
          - MDPI 2026 Regime-Aware LightGBM
          - QUANTA 2026 Regime-Filtered Hybrid
          - Hamilton 1989 HMM + Adams-MacKay 2007 变点检测

        Args:
            regime_data: RegimeDetectionSystem.update() 返回的状态字典
            default: regime_data 为空时的默认系数（向后兼容）

        Returns:
            增强系数 [0.0, 0.30]
        """
        if not regime_data:
            return default

        market_regime = regime_data.get("market_regime", "sideways")
        risk_level = regime_data.get("risk_level", "MEDIUM")
        is_transitioning = regime_data.get("is_transitioning", False)

        # 基础系数：趋势市场↑，震荡市场↓
        if market_regime in ("bull", "bear"):
            base = 0.20  # 趋势市场：趋势策略更有效
        elif market_regime == "sideways":
            base = 0.08  # 震荡市场：趋势策略效果差
        else:
            base = default  # 未知状态：保持默认

        # 风险等级衰减：越高越保守
        if risk_level == "EXTREME":
            base *= 0.0   # 极端风险：不增强
        elif risk_level == "HIGH":
            base *= 0.4   # 高风险：大幅衰减
        elif risk_level == "MEDIUM":
            base *= 0.7   # 中风险：温和衰减
        # LOW: 不衰减

        # 状态转换期：再乘 0.5（不确定性高）
        if is_transitioning:
            base *= 0.5

        # 钳制到合理范围 [0.0, 0.30]
        return max(0.0, min(0.30, base))

    def _assess_market_state(
        self, market: Any, indicators: Dict[str, float],
    ) -> Dict[str, Any]:
        """评估市场状态

        根据基因中的趋势检测器和波动检测器，判断当前市场是否适合交易。
        """
        close = market.close if hasattr(market, 'close') else market.get('close', 0)
        if not close:
            return {"suitable_for_trading": False, "regime": "unknown"}

        # 趋势判断
        trend_signals = []
        if "ema_cross" in self.gene.market_state_trend_detectors:
            ema12 = indicators.get("ema_12", close)
            ema26 = indicators.get("ema_26", close)
            if ema12 > ema26:
                trend_signals.append(1)  # 看涨
            else:
                trend_signals.append(-1)  # 看跌

        if "adx" in self.gene.market_state_trend_detectors:
            adx = indicators.get("adx_14", 25)
            if adx > 25:
                trend_signals.append(1)  # 趋势明确
            elif adx < 20:
                trend_signals.append(0)  # 震荡

        if "sma_cross" in self.gene.market_state_trend_detectors:
            sma20 = indicators.get("sma_20", close)
            sma50 = indicators.get("sma_50", close)
            if sma20 > sma50:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)

        # 波动判断
        vol_signals = []
        if "bollinger" in self.gene.market_state_volatility_detectors:
            upper = indicators.get("bollinger_upper", close * 1.05)
            lower = indicators.get("bollinger_lower", close * 0.95)
            bandwidth = (upper - lower) / close
            if bandwidth > 0.05:
                vol_signals.append("high")
            elif bandwidth < 0.02:
                vol_signals.append("low")
            else:
                vol_signals.append("normal")

        if "atr" in self.gene.market_state_volatility_detectors:
            atr = indicators.get("atr_14", 0)
            atr_pct = atr / close if close > 0 else 0
            if atr_pct > 0.03:
                vol_signals.append("high")
            elif atr_pct < 0.01:
                vol_signals.append("low")
            else:
                vol_signals.append("normal")

        # 综合判断
        avg_trend = sum(trend_signals) / len(trend_signals) if trend_signals else 0

        if "high" in vol_signals:
            regime = "volatile"
        elif "low" in vol_signals:
            regime = "quiet"
        elif avg_trend > 0.3:
            regime = "uptrend"
        elif avg_trend < -0.3:
            regime = "downtrend"
        else:
            regime = "ranging"

        # 世界观过滤：不是所有世界观都适合所有市场状态
        suitable = True
        wv = self.gene.worldview_primary
        # v3.32g: ranging+quiet市场严格过滤 (架构修复, 非过拟合)
        # 根因: 趋势策略在震荡市失效, quiet市场无趋势可跟随
        # 证据: v332f ranging 27笔11.1%胜率PnL=-12.81, quiet 3笔0%胜率PnL=-1.63
        #       v332c ranging 24笔45.8%胜率PnL=-3.05, quiet 5笔0%胜率PnL=-3.67
        # 修复: ranging/quiet市场只允许均值回归类世界观交易, 其他禁止
        # v332g验证: ranging 0笔 vs v332f 27笔, quiet 0笔 vs v332f 3笔 ✅过滤100%生效
        if regime == "ranging":
            if wv not in ("mean_reversion", "scalping", "mean_reversion_ml"):
                suitable = False
        elif regime == "quiet":
            if wv not in ("mean_reversion", "mean_reversion_ml"):
                suitable = False
        # v3.32h: downtrend市场过滤 (基于v332g新亏损源)
        # 证据: v332g volatile 34笔17.65%胜率PnL=-11.61(53.1%亏损)
        #       v332g uptrend 15笔6.67%胜率PnL=-7.82(35.8%亏损)
        #       v332g downtrend 6笔0%胜率PnL=-2.42(11.1%亏损)
        # 根因: downtrend市场做空被禁, 做多必亏
        # 修复v1(过严): volatile+uptrend+downtrend全过滤 → 只1笔交易(过滤过严)
        # 修复v2(当前): 只保留downtrend过滤, volatile/uptrend保留(有进化空间)
        #   - volatile 17.65%胜率非0%, 进化可优化
        #   - uptrend 6.67%胜率非0%, 进化可优化
        #   - downtrend 0%胜率 + 做空被禁 = 必亏, 必须过滤
        elif regime == "downtrend":
            # downtrend市场做空被禁, 做多必亏, 只允许均值回归抄底
            if wv not in ("mean_reversion", "mean_reversion_ml", "scalping"):
                suitable = False

        return {
            "suitable_for_trading": suitable,
            "regime": regime,
            "trend_strength": avg_trend,
            "volatility_regime": max(set(vol_signals), key=vol_signals.count) if vol_signals else "normal",
        }

