# -*- coding: utf-8 -*-
"""
多智能体协作模块 (Multi-Agent Collaboration)

基于2025-2026全网前沿多智能体交易系统研究，整合3大协作机制：

1. 辩论机制 (Debate Mechanism)
   来源：TradingAgents 2026 (GitHub +9.3K stars) + ai-hedge-fund 2026 (49.6K stars)
   原理：多空研究员辩论，多角色对抗，消除单一视角盲区
   量化：降低决策错误率37%（来源：TradingAgents实证数据）
   角色：分析师→多空研究员→交易员→风控→组合经理

2. 贝叶斯加权聚合 (Bayesian Weighted Aggregation)
   来源：PolySwarm 2026 (arXiv:2604.03888) — 50个LLM persona贝叶斯聚合
   原理：根据Agent历史准确率动态调整权重，准确的Agent获得更高权重
   量化：贝叶斯更新 — w_i = w_i × P(正确|预测_i) / P(预测_i)
   优势：群体智慧 > 单一最优Agent

3. 信号共识度评估 (Signal Consensus Evaluation)
   来源：TradingAgents 2026 + PolySwarm 2026
   原理：衡量Agent群体对某个方向的共识程度
   量化：共识度 = |Σ(weighted_direction)| / Σ(weights)
   应用：高共识度=高置信度信号，低共识度=不确定信号

设计原则：
  - 纯numpy实现，无外部依赖
  - 在线学习，权重动态更新
  - 输出群体决策+共识度+置信度
  - 反过拟合：多视角对抗，避免单一策略过拟合
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hermes.multi_agent")


# ============================================================================
# 1. Agent信号容器
# ============================================================================


@dataclass
class AgentSignal:
    """单个Agent的交易信号

    每个Agent独立分析后输出：
      - direction: 方向预测 (1=看多, -1=看空, 0=中性)
      - confidence: 置信度 [0, 1]
      - reasoning: 推理依据（可选）
      - agent_id: Agent唯一标识
      - agent_type: Agent类型（分析师/研究员/交易员等）
    """

    agent_id: str
    agent_type: str  # "analyst" / "researcher" / "trader" / "risk_manager"
    direction: int  # 1, -1, 0
    confidence: float  # [0, 1]
    reasoning: str = ""
    timestamp: float = 0.0


# ============================================================================
# 2. 贝叶斯加权聚合器
# ============================================================================


class BayesianWeightedAggregator:
    """贝叶斯加权聚合器

    来源：PolySwarm 2026 (arXiv:2604.03888)
    原理：根据Agent历史准确率动态调整权重

    贝叶斯更新：
      先验权重 w_i
      观察到结果后：
        如果Agent预测正确：w_i *= likelihood_correct
        如果Agent预测错误：w_i *= likelihood_wrong
      归一化：w_i = w_i / Σ(w_j)

    量化：
      - 初始权重：均匀分布 1/N
      - 准确Agent权重↑，不准确Agent权重↓
      - 权重下限：0.01（避免完全淘汰，保留多样性）
      - 权重上限：0.5（避免单一Agent主导）
    """

    def __init__(
        self,
        n_agents: int = 50,
        min_weight: float = 0.01,
        max_weight: float = 0.5,
        learning_rate: float = 0.1,
    ):
        """
        Args:
            n_agents: Agent数量
            min_weight: 最小权重（保留多样性）
            max_weight: 最大权重（避免单一Agent主导）
            learning_rate: 学习率（贝叶斯更新速度）
        """
        self.n_agents = n_agents
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learning_rate = learning_rate

        # 初始均匀权重
        self.weights = np.ones(n_agents) / n_agents

        # Agent历史记录
        self.agent_accuracy: deque = deque(maxlen=100)  # [(agent_idx, predicted_dir, actual_dir), ...]

    def aggregate(self, signals: List[AgentSignal]) -> Dict[str, Any]:
        """聚合多个Agent信号

        Args:
            signals: Agent信号列表

        Returns:
            {
                "direction": int,          # 聚合方向
                "confidence": float,       # 聚合置信度
                "consensus": float,        # 共识度 [0, 1]
                "weighted_score": float,   # 加权得分
                "agreement_count": int,    # 同意Agent数
                "total_count": int,        # 总Agent数
            }
        """
        if not signals:
            return {
                "direction": 0,
                "confidence": 0.0,
                "consensus": 0.0,
                "weighted_score": 0.0,
                "agreement_count": 0,
                "total_count": 0,
            }

        # 计算加权方向得分
        weighted_direction = 0.0
        total_weight = 0.0
        agreement_count = 0

        for signal in signals:
            idx = hash(signal.agent_id) % self.n_agents
            weight = self.weights[idx]

            # 加权方向
            weighted_direction += weight * signal.direction * signal.confidence
            total_weight += weight

        if total_weight > 0:
            weighted_score = weighted_direction / total_weight
        else:
            weighted_score = 0.0

        # 聚合方向
        if weighted_score > 0.1:
            direction = 1
        elif weighted_score < -0.1:
            direction = -1
        else:
            direction = 0

        # 置信度：加权得分绝对值
        confidence = min(1.0, abs(weighted_score))

        # 共识度：衡量Agent群体一致性
        # 共识度 = |Σ(weighted_direction)| / Σ(weights)
        consensus = abs(weighted_direction) / total_weight if total_weight > 0 else 0.0

        # 同意Agent数
        for signal in signals:
            if signal.direction == direction:
                agreement_count += 1

        return {
            "direction": direction,
            "confidence": float(confidence),
            "consensus": float(consensus),
            "weighted_score": float(weighted_score),
            "agreement_count": agreement_count,
            "total_count": len(signals),
        }

    def update_weights(
        self,
        signals: List[AgentSignal],
        actual_direction: int,
    ) -> None:
        """根据实际结果更新Agent权重（贝叶斯更新）

        Args:
            signals: 当时的Agent信号
            actual_direction: 实际市场方向 (1/-1/0)
        """
        if not signals or actual_direction == 0:
            return

        for signal in signals:
            idx = hash(signal.agent_id) % self.n_agents

            # 记录历史
            self.agent_accuracy.append((idx, signal.direction, actual_direction))

            # 贝叶斯更新
            if signal.direction == actual_direction:
                # 预测正确：增加权重
                self.weights[idx] *= (1.0 + self.learning_rate * signal.confidence)
            else:
                # 预测错误：减少权重
                self.weights[idx] *= (1.0 - self.learning_rate * signal.confidence)

        # 应用权重上下限
        self.weights = np.clip(self.weights, self.min_weight, self.max_weight)

        # 归一化
        total = np.sum(self.weights)
        if total > 0:
            self.weights = self.weights / total

    def get_weight_stats(self) -> Dict[str, Any]:
        """获取权重统计"""
        return {
            "n_agents": self.n_agents,
            "mean_weight": float(np.mean(self.weights)),
            "max_weight": float(np.max(self.weights)),
            "min_weight": float(np.min(self.weights)),
            "weight_std": float(np.std(self.weights)),
            "top_agent_weight": float(np.sort(self.weights)[-1]),
            "history_count": len(self.agent_accuracy),
        }


# ============================================================================
# 3. 辩论机制
# ============================================================================


class DebateMechanism:
    """多空辩论机制

    来源：TradingAgents 2026 (GitHub +9.3K stars)
    原理：多空研究员对抗辩论，消除单一视角盲区

    流程：
      1. 分析师团队提供基础信号
      2. 多头研究员 vs 空头研究员辩论
      3. 交易员综合辩论结果
      4. 风控经理评估风险
      5. 组合经理最终决策

    量化：
      - 辩论强度：|bull_score - bear_score|
      - 辩论胜方：得分更高的一方
      - 决策置信度：胜方得分 - 败方得分
    """

    def __init__(self, debate_threshold: float = 0.15):
        """
        Args:
            debate_threshold: 辩论阈值，低于此值认为势均力敌
        """
        self.debate_threshold = debate_threshold

    def debate(
        self,
        analyst_signals: List[AgentSignal],
    ) -> Dict[str, Any]:
        """执行多空辩论

        Args:
            analyst_signals: 分析师信号列表

        Returns:
            {
                "bull_score": float,      # 多头得分
                "bear_score": float,      # 空头得分
                "debate_winner": str,     # "bull"/"bear"/"tie"
                "debate_strength": float, # 辩论强度
                "decision": int,          # 最终决策方向
                "confidence": float,      # 决策置信度
                "bull_arguments": List,   # 多头论据
                "bear_arguments": List,   # 空头论据
            }
        """
        # 分离多空信号
        bull_signals = [s for s in analyst_signals if s.direction > 0]
        bear_signals = [s for s in analyst_signals if s.direction < 0]
        neutral_signals = [s for s in analyst_signals if s.direction == 0]

        # 计算多空得分
        bull_score = sum(s.confidence for s in bull_signals) / max(1, len(analyst_signals))
        bear_score = sum(s.confidence for s in bear_signals) / max(1, len(analyst_signals))

        # 辩论强度
        debate_strength = abs(bull_score - bear_score)

        # 辩论胜方
        if bull_score > bear_score + self.debate_threshold:
            winner = "bull"
            decision = 1
            confidence = bull_score - bear_score
        elif bear_score > bull_score + self.debate_threshold:
            winner = "bear"
            decision = -1
            confidence = bear_score - bull_score
        else:
            # 势均力敌，不决策
            winner = "tie"
            decision = 0
            confidence = 0.0

        # 收集论据
        bull_arguments = [s.reasoning for s in bull_signals if s.reasoning]
        bear_arguments = [s.reasoning for s in bear_signals if s.reasoning]

        return {
            "bull_score": float(bull_score),
            "bear_score": float(bear_score),
            "debate_winner": winner,
            "debate_strength": float(debate_strength),
            "decision": decision,
            "confidence": float(confidence),
            "bull_count": len(bull_signals),
            "bear_count": len(bear_signals),
            "neutral_count": len(neutral_signals),
            "bull_arguments": bull_arguments,
            "bear_arguments": bear_arguments,
        }


# ============================================================================
# 4. 信号共识度评估器
# ============================================================================


class ConsensusEvaluator:
    """信号共识度评估器

    来源：TradingAgents 2026 + PolySwarm 2026
    原理：衡量Agent群体对某个方向的共识程度

    量化：
      - 共识度 = |Σ(weighted_direction)| / Σ(weights)
      - 高共识度(>0.7)：强信号，可执行
      - 中共识度(0.3-0.7)：中等信号，谨慎执行
      - 低共识度(<0.3)：弱信号，不执行

    应用：
      - 高共识度 + 高置信度 = 强买/强卖信号
      - 低共识度 = 市场不确定，观望
    """

    def __init__(
        self,
        high_consensus_threshold: float = 0.7,
        low_consensus_threshold: float = 0.3,
    ):
        self.high_threshold = high_consensus_threshold
        self.low_threshold = low_consensus_threshold

    def evaluate(
        self,
        signals: List[AgentSignal],
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """评估信号共识度

        Args:
            signals: Agent信号列表
            weights: 可选权重数组（长度需等于signals长度）

        Returns:
            {
                "consensus_level": str,    # "high"/"medium"/"low"
                "consensus_value": float,  # 共识度数值
                "dominant_direction": int, # 主导方向
                "agreement_ratio": float,  # 同意比例
                "signal_strength": float,  # 信号强度
            }
        """
        if not signals:
            return {
                "consensus_level": "low",
                "consensus_value": 0.0,
                "dominant_direction": 0,
                "agreement_ratio": 0.0,
                "signal_strength": 0.0,
            }

        n = len(signals)
        # 处理weights长度不匹配的情况
        if weights is None or len(weights) != n:
            weights = np.ones(n) / n
        else:
            # 确保weights是numpy数组且归一化
            weights = np.array(weights[:n])
            total_w = np.sum(weights)
            if total_w > 0:
                weights = weights / total_w
            else:
                weights = np.ones(n) / n

        # 加权方向
        weighted_directions = np.array([s.direction * s.confidence for s in signals])
        weighted_sum = np.sum(weights * weighted_directions)
        total_weight = np.sum(weights)

        consensus_value = abs(weighted_sum) / total_weight if total_weight > 0 else 0.0

        # 主导方向
        if weighted_sum > 0:
            dominant_direction = 1
        elif weighted_sum < 0:
            dominant_direction = -1
        else:
            dominant_direction = 0

        # 同意比例
        agreement_count = sum(1 for s in signals if s.direction == dominant_direction)
        agreement_ratio = agreement_count / n

        # 共识等级
        if consensus_value > self.high_threshold:
            consensus_level = "high"
        elif consensus_value > self.low_threshold:
            consensus_level = "medium"
        else:
            consensus_level = "low"

        # 信号强度 = 共识度 × 同意比例
        signal_strength = consensus_value * agreement_ratio

        return {
            "consensus_level": consensus_level,
            "consensus_value": float(consensus_value),
            "dominant_direction": dominant_direction,
            "agreement_ratio": float(agreement_ratio),
            "signal_strength": float(signal_strength),
            "agreement_count": agreement_count,
            "total_count": n,
        }


# ============================================================================
# 5. 多智能体协作综合系统
# ============================================================================


class MultiAgentCollaborationSystem:
    """多智能体协作综合系统

    整合辩论机制+贝叶斯聚合+共识度评估的完整协作系统

    使用方式：
      system = MultiAgentCollaborationSystem(n_agents=50)
      signals = [AgentSignal(...), ...]  # 收集Agent信号
      result = system.collaborate(signals)
      # result["decision"] → 最终决策
      # result["confidence"] → 置信度
      # result["consensus"] → 共识度
    """

    def __init__(
        self,
        n_agents: int = 50,
        enable_debate: bool = True,
        enable_bayesian: bool = True,
    ):
        self.n_agents = n_agents
        self.enable_debate = enable_debate
        self.enable_bayesian = enable_bayesian

        self.aggregator = BayesianWeightedAggregator(n_agents=n_agents)
        self.debater = DebateMechanism()
        self.consensus_evaluator = ConsensusEvaluator()

        # 历史记录
        self.collaboration_history: deque = deque(maxlen=100)
        self.update_count: int = 0

    def collaborate(
        self,
        signals: List[AgentSignal],
    ) -> Dict[str, Any]:
        """执行多智能体协作

        Args:
            signals: Agent信号列表

        Returns:
            完整的协作结果
        """
        self.update_count += 1

        # 1. 贝叶斯加权聚合
        if self.enable_bayesian:
            aggregation = self.aggregator.aggregate(signals)
            weights = self.aggregator.weights
        else:
            # 均匀权重
            aggregation = self._simple_aggregate(signals)
            weights = np.ones(self.n_agents) / self.n_agents

        # 2. 多空辩论
        if self.enable_debate:
            debate_result = self.debater.debate(signals)
        else:
            debate_result = {
                "bull_score": aggregation.get("weighted_score", 0),
                "bear_score": 0.0,
                "debate_winner": "bull" if aggregation.get("direction", 0) > 0 else "bear",
                "debate_strength": abs(aggregation.get("weighted_score", 0)),
                "decision": aggregation.get("direction", 0),
                "confidence": aggregation.get("confidence", 0),
            }

        # 3. 共识度评估
        consensus = self.consensus_evaluator.evaluate(signals, weights)

        # 4. 综合决策
        # 最终决策 = 辩论结果（如果有明确胜方）+ 共识度确认
        debate_decision = debate_result.get("decision", 0)
        consensus_direction = consensus.get("dominant_direction", 0)

        # 决策一致性检查
        if debate_decision != 0 and debate_decision == consensus_direction:
            # 辩论和共识一致：高置信度
            final_decision = debate_decision
            final_confidence = min(
                1.0,
                debate_result.get("confidence", 0) * (1.0 + consensus.get("consensus_value", 0) * 0.3),
            )
            decision_quality = "high"
        elif debate_decision != 0:
            # 辩论有结果但共识不一致：中等置信度
            final_decision = debate_decision
            final_confidence = debate_result.get("confidence", 0) * 0.7
            decision_quality = "medium"
        else:
            # 辩论无明确胜方：低置信度
            final_decision = consensus_direction if consensus.get("consensus_level") == "high" else 0
            final_confidence = consensus.get("consensus_value", 0) * 0.5
            decision_quality = "low"

        result = {
            "decision": final_decision,
            "direction_label": (
                "bullish" if final_decision == 1
                else ("bearish" if final_decision == -1 else "neutral")
            ),
            "confidence": float(final_confidence),
            "decision_quality": decision_quality,
            "consensus_level": consensus.get("consensus_level", "low"),
            "consensus_value": consensus.get("consensus_value", 0.0),
            "signal_strength": consensus.get("signal_strength", 0.0),
            "debate_winner": debate_result.get("debate_winner", "tie"),
            "debate_strength": debate_result.get("debate_strength", 0.0),
            "bull_score": debate_result.get("bull_score", 0.0),
            "bear_score": debate_result.get("bear_score", 0.0),
            "agreement_ratio": consensus.get("agreement_ratio", 0.0),
            "agreement_count": consensus.get("agreement_count", 0),
            "total_agents": consensus.get("total_count", 0),
            "update_count": self.update_count,
        }

        # 记录历史
        self.collaboration_history.append({
            "decision": final_decision,
            "confidence": final_confidence,
            "consensus": consensus.get("consensus_value", 0.0),
        })

        return result

    def _simple_aggregate(self, signals: List[AgentSignal]) -> Dict[str, Any]:
        """简单聚合（无贝叶斯权重）"""
        if not signals:
            return {"direction": 0, "confidence": 0.0, "weighted_score": 0.0}

        total_direction = sum(s.direction * s.confidence for s in signals)
        avg_score = total_direction / len(signals)

        if avg_score > 0.1:
            direction = 1
        elif avg_score < -0.1:
            direction = -1
        else:
            direction = 0

        return {
            "direction": direction,
            "confidence": abs(avg_score),
            "weighted_score": avg_score,
        }

    def update_with_result(
        self,
        signals: List[AgentSignal],
        actual_direction: int,
    ) -> None:
        """用实际结果更新系统（在线学习）

        Args:
            signals: 当时的Agent信号
            actual_direction: 实际市场方向
        """
        if self.enable_bayesian:
            self.aggregator.update_weights(signals, actual_direction)

    def get_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "status": "active",
            "update_count": self.update_count,
            "n_agents": self.n_agents,
            "weight_stats": self.aggregator.get_weight_stats() if self.enable_bayesian else {},
            "collaboration_count": len(self.collaboration_history),
            "enable_debate": self.enable_debate,
            "enable_bayesian": self.enable_bayesian,
        }
