# -*- coding: utf-8 -*-
"""
llm_agent_trading.py — LLM Agent推理交易策略 v1.0

Phase 8.6: 从抽象世界观升级为具体策略实现

注意: 本策略不调用真实LLM (最小化依赖原则), 而是用技术指标
模拟ReAct+ToT+多Agent辩论的决策流程。

算法:
  1. ReAct 5步推理:
     Step1: 趋势 (MA短/长交叉)
     Step2: 动量 (MACD)
     Step3: RSI (超买超卖)
     Step4: 量能 (成交量比)
     Step5: 综合判断
  2. ToT 3分支并行:
     - Bull branch (bias=+0.3): 乐观解读
     - Bear branch (bias=-0.3): 悲观解读
     - Range branch (bias=0.0): 中性解读
  3. 多Agent辩论 (4 Agent):
     - Bull Agent: 偏多
     - Bear Agent: 偏空
     - Risk Agent: 风险否决 (drawdown > threshold)
     - Macro Agent: 宏观视角 (波动率+趋势)
  4. 反幻觉校验: 信号一致性检查
  5. 共识投票: 60%分支 + 40%Agent

信号类型:
  - LLM_LONG / LLM_SHORT / LLM_FLAT / LLM_RISK_VETO
"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LLMAgentTradingStrategy:
    """LLM Agent推理交易策略 (技术指标模拟版)

    模拟ReAct+ToT+多Agent辩论流程, 无需真实LLM调用。
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ma_short: int = 10,
        ma_long: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        react_steps: int = 5,
        n_branches: int = 3,
        branch_bull_bias: float = 0.3,
        branch_bear_bias: float = -0.3,
        n_agents: int = 4,
        debate_rounds: int = 3,
        agent_learning_rate: float = 0.1,
        consensus_threshold: float = 0.65,
        hallucination_threshold: float = 0.55,
        high_divergence_threshold: float = 0.55,
        min_history: int = 60,
        max_volatility_threshold: float = 0.06,
        risk_veto_drawdown: float = 0.15,
        **kwargs: Any,
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.react_steps = react_steps
        self.n_branches = n_branches
        self.branch_bull_bias = branch_bull_bias
        self.branch_bear_bias = branch_bear_bias
        self.n_agents = n_agents
        self.debate_rounds = debate_rounds
        self.agent_learning_rate = agent_learning_rate
        self.consensus_threshold = consensus_threshold
        self.hallucination_threshold = hallucination_threshold
        self.high_divergence_threshold = high_divergence_threshold
        self.min_history = min_history
        self.max_volatility_threshold = max_volatility_threshold
        self.risk_veto_drawdown = risk_veto_drawdown

        self._prices: deque = deque(maxlen=200)
        self._highs: deque = deque(maxlen=200)
        self._lows: deque = deque(maxlen=200)
        self._volumes: deque = deque(maxlen=200)
        self._peak_value: float = 0.0

        # Agent权重 (多/空/风控/宏观)
        self._agent_weights = np.array([0.25, 0.25, 0.25, 0.25])

    def update_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """Feeder: 接收价格"""
        self._prices.append(price)
        if price > self._peak_value:
            self._peak_value = price

    def update_high_low(self, high: float, low: float) -> None:
        """Feeder: 接收高低价"""
        self._highs.append(high)
        self._lows.append(low)

    def update_volume(self, volume: float) -> None:
        """Feeder: 接收成交量"""
        self._volumes.append(volume)

    def update_price_volume(
        self, price: float, volume: float, timestamp: Optional[float] = None
    ) -> None:
        """Feeder: 价格+成交量"""
        self.update_price(price, timestamp)
        self.update_volume(volume)

    def _compute_rsi(self) -> float:
        if len(self._prices) < self.rsi_period + 1:
            return 50.0
        prices = list(self._prices)[-self.rsi_period - 1:]
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_macd(self) -> Tuple[float, float, float]:
        if len(self._prices) < self.macd_slow + self.macd_signal:
            return (0.0, 0.0, 0.0)
        prices = list(self._prices)
        ema_fast = self._ema(prices, self.macd_fast)
        ema_slow = self._ema(prices, self.macd_slow)
        macd_line = ema_fast - ema_slow
        # signal line
        macd_history = []
        for i in range(self.macd_signal, 0, -1):
            if len(prices) > self.macd_slow + i:
                seg = prices[:-(i)]
                ef = self._ema(seg, self.macd_fast)
                es = self._ema(seg, self.macd_slow)
                macd_history.append(ef - es)
        if not macd_history:
            macd_history.append(macd_line)
        signal_line = float(np.mean(macd_history[-self.macd_signal:]))
        histogram = macd_line - signal_line
        return (float(macd_line), signal_line, float(histogram))

    @staticmethod
    def _ema(data: List[float], period: int) -> float:
        if len(data) < period:
            return float(np.mean(data)) if data else 0.0
        alpha = 2.0 / (period + 1)
        ema = float(data[0])
        for p in data[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def _compute_ma(self, period: int) -> float:
        if len(self._prices) < period:
            return float(np.mean(list(self._prices))) if self._prices else 0.0
        return float(np.mean(list(self._prices)[-period:]))

    def _compute_bollinger(self) -> Tuple[float, float, float]:
        if len(self._prices) < self.bb_period:
            p = float(np.mean(list(self._prices))) if self._prices else 0.0
            return (p, p, p)
        recent = list(self._prices)[-self.bb_period:]
        ma = float(np.mean(recent))
        std = float(np.std(recent))
        return (ma + self.bb_std * std, ma, ma - self.bb_std * std)

    def _compute_atr(self) -> float:
        if len(self._highs) < self.atr_period or len(self._lows) < self.atr_period:
            return 0.0
        trs = []
        for i in range(-self.atr_period, 0):
            if abs(i) <= len(self._highs) and abs(i) <= len(self._lows) and abs(i) <= len(self._prices):
                h = self._highs[i]
                l = self._lows[i]
                prev_c = self._prices[i - 1] if abs(i - 1) <= len(self._prices) else l
                tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                trs.append(tr)
        return float(np.mean(trs)) if trs else 0.0

    def _compute_volume_ratio(self) -> float:
        if len(self._volumes) < 20:
            return 1.0
        recent = list(self._volumes)[-5:]
        avg = float(np.mean(list(self._volumes)[-20:-5]))
        return float(np.mean(recent) / avg) if avg > 0 else 1.0

    def _react_reasoning(self) -> List[Tuple[str, float]]:
        """ReAct 5步推理, 返回[(step_name, score)]"""
        steps = []
        rsi = self._compute_rsi()
        macd, signal, hist = self._compute_macd()
        ma_s = self._compute_ma(self.ma_short)
        ma_l = self._compute_ma(self.ma_long)
        vol_ratio = self._compute_volume_ratio()

        # Step1: 趋势
        trend_score = (ma_s - ma_l) / ma_l if ma_l > 0 else 0.0
        steps.append(("trend", float(np.clip(trend_score * 10, -1, 1))))

        # Step2: 动量
        steps.append(("momentum", float(np.clip(hist * 100, -1, 1))))

        # Step3: RSI (50为中性)
        rsi_score = (rsi - 50) / 50.0
        steps.append(("rsi", float(np.clip(rsi_score, -1, 1))))

        # Step4: 量能
        vol_score = float(np.clip((vol_ratio - 1.0) * 2, -1, 1))
        steps.append(("volume", vol_score))

        # Step5: 综合
        composite = float(np.mean([s for _, s in steps[:4]]))
        steps.append(("composite", composite))
        return steps

    def _tot_branches(self, composite: float) -> List[Tuple[str, float]]:
        """ToT 3分支并行推理"""
        branches = []
        # Bull branch: 乐观解读
        bull_score = float(np.clip(composite + self.branch_bull_bias, -1, 1))
        branches.append(("bull", bull_score))
        # Bear branch: 悲观解读
        bear_score = float(np.clip(composite + self.branch_bear_bias, -1, 1))
        branches.append(("bear", bear_score))
        # Range branch: 中性解读
        range_score = float(np.clip(composite * 0.5, -1, 1))
        branches.append(("range", range_score))
        return branches

    def _agent_debate(self, composite: float, branches: List[Tuple[str, float]]) -> Tuple[float, bool]:
        """多Agent辩论, 返回(consensus_score, risk_veto)"""
        rsi = self._compute_rsi()
        atr = self._compute_atr()
        current_price = self._prices[-1] if self._prices else 0.0
        drawdown = (self._peak_value - current_price) / self._peak_value if self._peak_value > 0 else 0.0

        # Agent 1: Bull (偏多)
        bull_agent = composite * 0.7 + (1 if rsi < self.rsi_oversold else 0) * 0.3
        # Agent 2: Bear (偏空)
        bear_agent = -composite * 0.7 - (1 if rsi > self.rsi_overbought else 0) * 0.3
        # Agent 3: Risk (风控否决)
        risk_veto = drawdown > self.risk_veto_drawdown or atr / current_price > self.max_volatility_threshold
        risk_agent = 0.0 if risk_veto else composite * 0.3
        # Agent 4: Macro (宏观波动率视角)
        vol = atr / current_price if current_price > 0 else 0.0
        macro_agent = -0.3 if vol > self.max_volatility_threshold else composite * 0.5

        # 辩论: 多轮贝叶斯权重更新
        agent_signals = np.array([bull_agent, bear_agent, risk_agent, macro_agent])
        for _ in range(self.debate_rounds):
            # 权重更新: 表现好的agent权重增加
            consensus_dir = np.sign(np.average(agent_signals, weights=self._agent_weights))
            for i in range(len(self._agent_weights)):
                if np.sign(agent_signals[i]) == consensus_dir:
                    self._agent_weights[i] *= (1 + self.agent_learning_rate)
                else:
                    self._agent_weights[i] *= (1 - self.agent_learning_rate * 0.5)
            self._agent_weights = self._agent_weights / np.sum(self._agent_weights)

        consensus = float(np.average(agent_signals, weights=self._agent_weights))
        return consensus, bool(risk_veto)

    def _hallucination_check(self, react_scores: List[Tuple[str, float]], consensus: float) -> float:
        """反幻觉校验: 检查信号一致性"""
        scores = [s for _, s in react_scores[:-1]]  # 排除composite
        # 一致性 = 信号方向一致性
        signs = [1 if s > 0.1 else (-1 if s < -0.1 else 0) for s in scores]
        non_zero = [s for s in signs if s != 0]
        if not non_zero:
            return 0.5  # 信号不足
        consistency = abs(sum(non_zero)) / len(non_zero)
        return float(consistency)

    def analyze(self) -> Dict[str, Any]:
        """主分析函数"""
        result: Dict[str, Any] = {
            "signal_type": "LLM_FLAT",
            "direction": "flat",
            "confidence": 0.0,
            "strength": 0.0,
            "description": "",
        }

        if len(self._prices) < self.min_history:
            result["description"] = f"数据不足 {len(self._prices)}/{self.min_history}"
            return result

        # ReAct 5步推理
        react_steps = self._react_reasoning()
        composite = react_steps[-1][1]

        # ToT 3分支
        branches = self._tot_branches(composite)

        # 多Agent辩论
        consensus, risk_veto = self._agent_debate(composite, branches)

        # 反幻觉校验
        hallucination_score = self._hallucination_check(react_steps, consensus)
        if hallucination_score < self.hallucination_threshold:
            result.update({
                "signal_type": "LLM_FLAT",
                "direction": "flat",
                "description": f"反幻觉校验失败 consistency={hallucination_score:.2f}",
            })
            return result

        # 风控否决
        if risk_veto:
            result.update({
                "signal_type": "LLM_RISK_VETO",
                "direction": "flat",
                "confidence": 0.8,
                "strength": 0.8,
                "description": "风控Agent否决 (drawdown过高或波动率过大)",
            })
            return result

        # 分支分歧过大
        branch_scores = [s for _, s in branches]
        divergence = float(np.std(branch_scores))
        if divergence > self.high_divergence_threshold:
            result["description"] = f"分支分歧过大 div={divergence:.2f}"
            return result

        # 共识投票: 60%分支 + 40%Agent
        branch_consensus = float(np.mean(branch_scores))
        final_score = 0.6 * branch_consensus + 0.4 * consensus

        if final_score > self.consensus_threshold - 0.5:  # 调整阈值
            result.update({
                "signal_type": "LLM_LONG",
                "direction": "long",
                "confidence": min(1.0, abs(final_score)),
                "strength": abs(final_score),
                "description": f"ReAct+ToT+Agent共识做多 score={final_score:.3f}",
            })
        elif final_score < -(self.consensus_threshold - 0.5):
            result.update({
                "signal_type": "LLM_SHORT",
                "direction": "short",
                "confidence": min(1.0, abs(final_score)),
                "strength": abs(final_score),
                "description": f"ReAct+ToT+Agent共识做空 score={final_score:.3f}",
            })
        else:
            result["description"] = f"共识不足 score={final_score:.3f}"

        return result
