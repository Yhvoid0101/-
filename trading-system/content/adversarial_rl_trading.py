# -*- coding: utf-8 -*-
"""
对抗性强化学习交易引擎 (Adversarial RL Trading) — v3.4 新增

引入 2026 最新前沿 RL 技术,提升策略鲁棒性。

学术来源:
  - ArchetypeTrader (AAAI 2026): VQ-VAE 战略原型蒸馏 + RL 选择 + Policy Adapter
  - FineFT (KDD 2026): 集成Q学习器选择性TD更新 + VAE 能力边界检测 + 保守策略回退
  - TraderBench (ICLR 2026): 4级渐进对抗变换(baseline→noisy→meta→adversarial)

三大核心机制:
  1. **战略原型蒸馏** — 用 VQ-VAE 将历史轨迹蒸馏为 K 个离散战略原型码本,
     RL agent 学习在当前市场状态下选择最合适的原型
  2. **能力边界检测** — 训练 VAE 重建市场状态, 重建误差高的状态被视为超出能力边界,
     自动切换保守策略 (FineFT 降低 40%+ 风险)
  3. **对抗性测试** — 4级渐进扰动模拟市场操纵/对抗攻击, 策略必须在所有级别保持鲁棒

信号类型 (AdversarialRLSignalType):
  ARCHETYPE_MATCH         — 战略原型匹配(高置信度执行)
  ARCHETYPE_REFINE        — 原型精炼(Policy Adapter 调整)
  CAPABILITY_BOUNDARY     — 能力边界预警(切换保守)
  ADVERSARIAL_ALARM       — 对抗性攻击告警
  ENSEMBE_DISAGREE        — 集成Q学习器分歧(降低仓位)
  STRONG_BUY              — 强买(原型匹配+低对抗+能力内)
  STRONG_SELL             — 强卖
  HOLD                    — 持仓观望
  NEUTRAL                 — 中性

纯 numpy 实现, 无 PyTorch 依赖, 沙盘中可即时进化。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# 枚举定义
# ============================================================================

class AdversarialRLSignalType(Enum):
    """对抗性RL信号类型"""
    ARCHETYPE_MATCH = "archetype_match"             # 战略原型匹配
    ARCHETYPE_REFINE = "archetype_refine"           # 原型精炼
    CAPABILITY_BOUNDARY = "capability_boundary"     # 能力边界预警
    ADVERSARIAL_ALARM = "adversarial_alarm"         # 对抗性攻击告警
    ENSEMBLE_DISAGREE = "ensemble_disagree"         # 集成分歧
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    HOLD = "hold"
    NEUTRAL = "neutral"


class AdversarialLevel(Enum):
    """对抗性等级 (TraderBench 4级渐进)"""
    BASELINE = 0          # 基线: 真实历史数据
    NOISY = 1             # 噪声: 加入高斯噪声(模拟数据污染)
    META = 2              # 元: 加入相关性破坏(模拟市场结构变化)
    ADVERSARIAL = 3       # 对抗: PGD 风格定向扰动(模拟恶意攻击)


class ActionType(Enum):
    """交易动作"""
    FLAT = 0
    LONG_SMALL = 1
    LONG_MEDIUM = 2
    LONG_LARGE = 3
    SHORT_SMALL = 4
    SHORT_MEDIUM = 5
    SHORT_LARGE = 6


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StrategyArchetype:
    """战略原型 (ArchetypeTrader VQ codebook entry)"""
    archetype_id: int
    code: np.ndarray              # 原型编码向量
    label: str                    # 人类可读标签(如 "trend_following_bull")
    usage_count: int = 0          # 使用次数
    avg_reward: float = 0.0       # 平均回报
    last_used_step: int = 0


@dataclass
class AdversarialRLOpportunity:
    """对抗性RL机会"""
    signal_type: AdversarialRLSignalType
    selected_archetype: Optional[int]      # 选中的原型 ID
    action: ActionType
    q_value: float                         # Q 值估计
    capability_score: float                # 能力内分数 [0,1]
    adversarial_robustness: float          # 对抗鲁棒性 [0,1]
    ensemble_consensus: float              # 集成一致性 [0,1]
    conviction: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdversarialRLSignal:
    """对抗性RL综合信号"""
    signal_type: AdversarialRLSignalType
    direction: int                          # +1 多 / -1 空 / 0 观望
    conviction: float
    selected_archetype: Optional[int]
    action: ActionType
    adversarial_level: AdversarialLevel     # 当前对抗等级
    capability_score: float
    ensemble_consensus: float
    opportunities: List[AdversarialRLOpportunity] = field(default_factory=list)


# ============================================================================
# VQ-VAE 简化实现 (战略原型蒸馏)
# ============================================================================

class VQCodebook:
    """
    Vector Quantization 码本 (ArchetypeTrader Phase 1)

    将连续市场状态蒸馏为离散战略原型。
    在线 k-means 风格更新, 无需预训练。
    """

    def __init__(self, k: int = 8, dim: int = 16, lr: float = 0.01):
        self.k = k
        self.dim = dim
        self.lr = lr
        # 初始化码本: k 个原型, 每个 dim 维
        self.codes: np.ndarray = np.random.randn(k, dim) * 0.1
        self.usage_count: np.ndarray = np.zeros(k, dtype=int)
        self.label_map: Dict[int, str] = {
            0: "trend_bull",
            1: "trend_bear",
            2: "range_tight",
            3: "range_wide",
            4: "breakout_up",
            5: "breakout_down",
            6: "crisis_mode",
            7: "accumulation",
        }

    def quantize(self, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        量化输入向量

        Returns:
            (selected_index, distance, quantized_vector)
        """
        if x.shape[0] != self.dim:
            # 维度对齐
            if x.shape[0] > self.dim:
                x = x[:self.dim]
            else:
                x = np.resize(x, self.dim)

        # 计算到所有原型的距离
        diff = self.codes - x[np.newaxis, :]
        distances = np.sum(diff * diff, axis=1)
        selected = int(np.argmin(distances))
        min_dist = float(distances[selected])

        # 在线更新 (k-means 风格)
        self.codes[selected] = (1 - self.lr) * self.codes[selected] + self.lr * x
        self.usage_count[selected] += 1

        return selected, min_dist, self.codes[selected].copy()

    def get_archetype(self, idx: int) -> StrategyArchetype:
        """获取原型对象"""
        label = self.label_map.get(idx, f"archetype_{idx}")
        return StrategyArchetype(
            archetype_id=idx,
            code=self.codes[idx].copy(),
            label=label,
            usage_count=int(self.usage_count[idx]),
        )


# ============================================================================
# VAE 简化实现 (能力边界检测)
# ============================================================================

class SimplifiedVAE:
    """
    简化版 VAE (FineFT Stage II)

    用于检测市场状态是否在 agent 能力边界内。
    重建误差高 = 超出能力边界 = 切换保守策略。
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 8, lr: float = 0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        # Encoder: input -> hidden -> (mu, logvar)
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_mu = np.zeros(hidden_dim)
        self.W_logvar = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b_logvar = np.zeros(hidden_dim)
        # Decoder: hidden -> input
        self.W_dec = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_dec = np.zeros(input_dim)
        # 重建误差历史(用于动态阈值)
        self.recon_error_history: deque = deque(maxlen=200)
        self.error_threshold: float = 1.0  # 动态调整

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(self.W_enc1.T @ x + self.b_enc1)
        mu = self.W_mu.T @ h + self.b_mu
        logvar = self.W_logvar.T @ h + self.b_logvar
        # clip 防止数值不稳定
        logvar = np.clip(logvar, -4.0, 4.0)
        return mu, logvar

    def _reparametrize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def _decode(self, z: np.ndarray) -> np.ndarray:
        return self.W_dec.T @ z + self.b_dec

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """前向传播, 返回 (重建, 重建误差)"""
        if x.shape[0] != self.input_dim:
            if x.shape[0] > self.input_dim:
                x = x[:self.input_dim]
            else:
                x = np.resize(x, self.input_dim)

        mu, logvar = self._encode(x)
        z = self._reparametrize(mu, logvar)
        recon = self._decode(z)
        # 重建误差 (MSE)
        error = float(np.mean((recon - x) ** 2))
        return recon, error

    def update(self, x: np.ndarray) -> float:
        """
        在线更新 VAE, 返回重建误差

        使用简化的 ELBO 梯度下降
        """
        if x.shape[0] != self.input_dim:
            if x.shape[0] > self.input_dim:
                x = x[:self.input_dim]
            else:
                x = np.resize(x, self.input_dim)

        recon, error = self.forward(x)
        mu, logvar = self._encode(x)

        # 简化梯度: 朝向减小重建误差方向更新
        grad_recon = 2 * (recon - x) / self.input_dim  # dL/drecon

        # Update decoder
        z = self._reparametrize(mu, logvar)
        self.W_dec -= self.lr * np.outer(z, grad_recon)
        self.b_dec -= self.lr * grad_recon

        # Update encoder (简化: 只更新 W_enc1)
        grad_hidden = self.W_dec @ grad_recon  # 传播回 hidden
        grad_hidden *= (1 - np.tanh(self.W_enc1.T @ x + self.b_enc1) ** 2)
        self.W_enc1 -= self.lr * np.outer(x, grad_hidden)
        self.b_enc1 -= self.lr * grad_hidden

        # 记录误差
        self.recon_error_history.append(error)
        # 动态阈值 = 历史误差的 90 分位
        if len(self.recon_error_history) >= 50:
            arr = np.array(self.recon_error_history)
            self.error_threshold = float(np.percentile(arr, 90))

        return error

    def is_in_capability(self, x: np.ndarray) -> Tuple[bool, float]:
        """判断是否在能力边界内"""
        _, error = self.forward(x)
        return error <= self.error_threshold, error / max(self.error_threshold, 1e-6)


# ============================================================================
# 集成 Q 学习器 (FineFT Stage I)
# ============================================================================

class EnsembleQLearner:
    """
    集成 Q 学习器 (FineFT Stage I)

    多个 Q 网络通过选择性 TD 更新提升收敛性。
    """

    def __init__(self, n_learners: int = 4, state_dim: int = 16,
                 n_actions: int = 7, lr: float = 0.01, gamma: float = 0.95):
        self.n_learners = n_learners
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        # 每个 learner 一个简化的线性 Q 函数: Q(s,a) = W_a · s + b_a
        self.W: List[np.ndarray] = [
            np.random.randn(state_dim, n_actions) * 0.05
            for _ in range(n_learners)
        ]
        self.b: List[np.ndarray] = [
            np.zeros(n_actions) for _ in range(n_learners)
        ]
        # 每个 learner 的累计 TD 误差 (用于选择性更新)
        self.td_errors: List[float] = [0.0] * n_learners

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        集成预测, 返回 (平均Q值, 一致性分数)

        一致性分数: 各 learner 推荐动作的一致程度 [0, 1]
        """
        if state.shape[0] != self.state_dim:
            if state.shape[0] > self.state_dim:
                state = state[:self.state_dim]
            else:
                state = np.resize(state, self.state_dim)

        q_values = []
        actions = []
        for i in range(self.n_learners):
            q = self.W[i].T @ state + self.b[i]
            q_values.append(q)
            actions.append(int(np.argmax(q)))

        avg_q = np.mean(q_values, axis=0)
        # 一致性: 多少 learner 选择同一个动作
        most_common = max(set(actions), key=actions.count)
        consensus = actions.count(most_common) / self.n_learners

        return avg_q, consensus

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray) -> None:
        """
        选择性 TD 更新 (FineFT 核心)

        只更新 TD 误差低于中位数的 learner, 提升收敛性
        """
        if state.shape[0] != self.state_dim:
            state = state[:self.state_dim] if state.shape[0] > self.state_dim else np.resize(state, self.state_dim)
        if next_state.shape[0] != self.state_dim:
            next_state = next_state[:self.state_dim] if next_state.shape[0] > self.state_dim else np.resize(next_state, self.state_dim)

        # 计算每个 learner 的 TD 误差
        td_list = []
        for i in range(self.n_learners):
            q_curr = (self.W[i].T @ state + self.b[i])[action]
            q_next = np.max(self.W[i].T @ next_state + self.b[i])
            td = reward + self.gamma * q_next - q_curr
            td_list.append(td)
            self.td_errors[i] = 0.9 * self.td_errors[i] + 0.1 * abs(td)

        # 选择性更新: 只更新 TD 误差低于中位数的 learner
        median_td = float(np.median([abs(t) for t in td_list]))
        for i in range(self.n_learners):
            if abs(td_list[i]) <= median_td:
                # 梯度下降
                grad = -td_list[i] * state
                self.W[i][:, action] -= self.lr * grad
                self.b[i][action] -= self.lr * (-td_list[i])


# ============================================================================
# 主类: AdversarialRLTrading
# ============================================================================

class AdversarialRLTrading:
    """
    对抗性强化学习交易引擎

    融合三大 2026 前沿:
      1. ArchetypeTrader VQ 原型蒸馏
      2. FineFT 集成Q学习 + VAE 能力边界
      3. TraderBench 4级对抗测试
    """

    # 类常量
    STATE_DIM = 16
    N_ARCHETYPES = 8
    N_Q_LEARNERS = 4
    N_ACTIONS = 7

    CAPABILITY_LOW_THRESHOLD = 0.4     # 能力分数低于此值 → 保守
    ADVERSARIAL_ALARM_THRESHOLD = 0.5  # 对抗鲁棒性低于此值 → 告警
    ENSEMBLE_CONSENSUS_LOW = 0.5       # 集成一致性低于此值 → 降低仓位
    Q_VALUE_STRONG_THRESHOLD = 0.5     # Q 值强信号阈值

    ADVERSARIAL_NOISE_LEVELS = {
        AdversarialLevel.BASELINE: 0.0,
        AdversarialLevel.NOISY: 0.05,
        AdversarialLevel.META: 0.10,
        AdversarialLevel.ADVERSARIAL: 0.20,
    }

    HISTORY_WINDOW = 100
    TRAJECTORY_BUFFER_SIZE = 500

    def __init__(self):
        # VQ 码本 (战略原型)
        self.vq_codebook = VQCodebook(
            k=self.N_ARCHETYPES, dim=self.STATE_DIM, lr=0.01)

        # VAE (能力边界检测)
        self.vae = SimplifiedVAE(
            input_dim=self.STATE_DIM, hidden_dim=8, lr=0.005)

        # 集成 Q 学习器
        self.ensemble_q = EnsembleQLearner(
            n_learners=self.N_Q_LEARNERS, state_dim=self.STATE_DIM,
            n_actions=self.N_ACTIONS, lr=0.01, gamma=0.95)

        # 当前状态
        self.current_state: Optional[np.ndarray] = None
        self.current_archetype: Optional[int] = None
        self.current_action: ActionType = ActionType.FLAT
        self.current_adversarial_level: AdversarialLevel = AdversarialLevel.BASELINE

        # 历史轨迹
        self.trajectory: deque = deque(maxlen=self.TRAJECTORY_BUFFER_SIZE)
        self.state_history: deque = deque(maxlen=self.HISTORY_WINDOW)

        # 原型性能追踪
        self.archetype_rewards: Dict[int, deque] = {
            i: deque(maxlen=50) for i in range(self.N_ARCHETYPES)
        }

        # 对抗性测试结果
        self.adversarial_robustness: float = 1.0
        self.capability_score: float = 1.0
        self.ensemble_consensus: float = 1.0

        # 统计
        self.signal_count: int = 0
        self.capability_violations: int = 0
        self.adversarial_alarms: int = 0

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_state(self, features: List[float]) -> None:
        """更新市场状态"""
        if len(features) < self.STATE_DIM:
            features = list(features) + [0.0] * (self.STATE_DIM - len(features))
        elif len(features) > self.STATE_DIM:
            features = features[:self.STATE_DIM]

        self.current_state = np.array(features, dtype=float)
        self.state_history.append(self.current_state.copy())

        # 1. VQ 量化 (原型选择)
        self.current_archetype, _, _ = self.vq_codebook.quantize(self.current_state)

        # 2. VAE 在线更新 + 能力检测
        in_capability, score = self.vae.is_in_capability(self.current_state)
        self.capability_score = max(0.0, 1.0 - score / 2.0)
        if not in_capability:
            self.capability_violations += 1
        self.vae.update(self.current_state)

        # 3. 对抗性测试 (4级)
        self._run_adversarial_tests()

    def record_step(self, features: List[float], action: int, reward: float) -> None:
        """记录一步轨迹, 触发 Q 学习"""
        if len(features) < self.STATE_DIM:
            features = list(features) + [0.0] * (self.STATE_DIM - len(features))
        elif len(features) > self.STATE_DIM:
            features = features[:self.STATE_DIM]

        state = np.array(features, dtype=float)
        if len(self.trajectory) > 0:
            prev_state, prev_action, _ = self.trajectory[-1]
            # 集成 Q 学习器选择性更新
            self.ensemble_q.update(prev_state, prev_action, reward, state)

        self.trajectory.append((state, action, reward))

        # 更新原型性能
        if self.current_archetype is not None:
            self.archetype_rewards[self.current_archetype].append(reward)
            arch = self.vq_codebook.get_archetype(self.current_archetype)
            arch.usage_count += 1
            if len(self.archetype_rewards[self.current_archetype]) > 0:
                rewards = list(self.archetype_rewards[self.current_archetype])
                arch.avg_reward = sum(rewards) / len(rewards)

    # ------------------------------------------------------------------
    # 对抗性测试 (TraderBench 4级)
    # ------------------------------------------------------------------

    def _generate_adversarial_perturbation(self, x: np.ndarray,
                                           level: AdversarialLevel) -> np.ndarray:
        """
        生成对抗性扰动 (PGD 风格简化版)

        level: 0=baseline, 1=noisy, 2=meta, 3=adversarial
        """
        noise_scale = self.ADVERSARIAL_NOISE_LEVELS[level]
        if noise_scale == 0:
            return x.copy()

        if level == AdversarialLevel.NOISY:
            # 高斯噪声 (数据污染模拟)
            return x + np.random.randn(*x.shape) * noise_scale

        elif level == AdversarialLevel.META:
            # 相关性破坏 (市场结构变化模拟)
            perturbed = x.copy()
            # 随机打乱部分维度
            n_shuffle = max(1, int(len(x) * noise_scale))
            indices = np.random.choice(len(x), n_shuffle, replace=False)
            perturbed[indices] = -perturbed[indices]
            return perturbed

        elif level == AdversarialLevel.ADVERSARIAL:
            # PGD 风格定向扰动 (恶意攻击模拟)
            # 朝 Q 函数梯度方向扰动 (简化)
            grad = self._estimate_q_gradient(x)
            return x + noise_scale * np.sign(grad)

        return x.copy()

    def _estimate_q_gradient(self, x: np.ndarray) -> np.ndarray:
        """估计 Q 函数对输入的梯度 (简化)"""
        # 使用集成 Q 的平均梯度
        if self.current_state is None:
            return np.zeros_like(x)

        avg_q, _ = self.ensemble_q.predict(x)
        # 简化: 用数值差分
        grad = np.zeros_like(x)
        eps = 1e-4
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            q_plus, _ = self.ensemble_q.predict(x_plus)
            q_minus, _ = self.ensemble_q.predict(x_minus)
            grad[i] = (np.max(q_plus) - np.max(q_minus)) / (2 * eps)
        return grad

    def _run_adversarial_tests(self) -> None:
        """运行 4 级对抗性测试"""
        if self.current_state is None:
            return

        # 基线 Q 值
        baseline_q, _ = self.ensemble_q.predict(self.current_state)
        baseline_action = int(np.argmax(baseline_q))

        # 测试每个对抗级别
        consistent_actions = 0
        for level in [AdversarialLevel.NOISY, AdversarialLevel.META,
                      AdversarialLevel.ADVERSARIAL]:
            perturbed = self._generate_adversarial_perturbation(
                self.current_state, level)
            q, _ = self.ensemble_q.predict(perturbed)
            if int(np.argmax(q)) == baseline_action:
                consistent_actions += 1

        # 鲁棒性 = 一致动作比例
        self.adversarial_robustness = consistent_actions / 3.0

        # 自动升级对抗级别
        if self.adversarial_robustness < 0.33:
            self.current_adversarial_level = AdversarialLevel.ADVERSARIAL
            self.adversarial_alarms += 1
        elif self.adversarial_robustness < 0.66:
            self.current_adversarial_level = AdversarialLevel.META
        elif self.adversarial_robustness < 0.90:
            self.current_adversarial_level = AdversarialLevel.NOISY
        else:
            self.current_adversarial_level = AdversarialLevel.BASELINE

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def analyze(self) -> AdversarialRLSignal:
        """综合分析并生成信号"""
        if self.current_state is None:
            return AdversarialRLSignal(
                signal_type=AdversarialRLSignalType.NEUTRAL,
                direction=0, conviction=0.0,
                selected_archetype=None,
                action=ActionType.FLAT,
                adversarial_level=AdversarialLevel.BASELINE,
                capability_score=0.0,
                ensemble_consensus=0.0,
            )

        # 集成 Q 预测
        avg_q, consensus = self.ensemble_q.predict(self.current_state)
        self.ensemble_consensus = consensus
        best_action = ActionType(int(np.argmax(avg_q)))
        best_q = float(np.max(avg_q))

        # 决定信号
        opportunities = []

        # 1. 能力边界检测
        if self.capability_score < self.CAPABILITY_LOW_THRESHOLD:
            opportunities.append(AdversarialRLOpportunity(
                signal_type=AdversarialRLSignalType.CAPABILITY_BOUNDARY,
                selected_archetype=self.current_archetype,
                action=ActionType.FLAT,
                q_value=best_q,
                capability_score=self.capability_score,
                adversarial_robustness=self.adversarial_robustness,
                ensemble_consensus=consensus,
                conviction=1.0 - self.capability_score,
                metadata={'reason': 'capability_violation'}
            ))

        # 2. 对抗性告警
        if self.adversarial_robustness < self.ADVERSARIAL_ALARM_THRESHOLD:
            opportunities.append(AdversarialRLOpportunity(
                signal_type=AdversarialRLSignalType.ADVERSARIAL_ALARM,
                selected_archetype=self.current_archetype,
                action=ActionType.FLAT,
                q_value=best_q,
                capability_score=self.capability_score,
                adversarial_robustness=self.adversarial_robustness,
                ensemble_consensus=consensus,
                conviction=1.0 - self.adversarial_robustness,
                metadata={'level': self.current_adversarial_level.value}
            ))

        # 3. 集成分歧
        elif consensus < self.ENSEMBLE_CONSENSUS_LOW:
            opportunities.append(AdversarialRLOpportunity(
                signal_type=AdversarialRLSignalType.ENSEMBLE_DISAGREE,
                selected_archetype=self.current_archetype,
                action=best_action,
                q_value=best_q,
                capability_score=self.capability_score,
                adversarial_robustness=self.adversarial_robustness,
                ensemble_consensus=consensus,
                conviction=1.0 - consensus,
                metadata={'consensus': consensus}
            ))

        # 4. 原型匹配 (正常路径)
        else:
            # 强信号: Q 值高 + 能力内 + 对抗鲁棒
            if (abs(best_q) > self.Q_VALUE_STRONG_THRESHOLD and
                self.capability_score > 0.7 and
                self.adversarial_robustness > 0.7):
                signal_type = (AdversarialRLSignalType.STRONG_BUY
                               if best_q > 0 else
                               AdversarialRLSignalType.STRONG_SELL)
                conviction = min(1.0, abs(best_q) * self.capability_score *
                                 self.adversarial_robustness * consensus)
            else:
                # 普通原型匹配
                signal_type = AdversarialRLSignalType.ARCHETYPE_MATCH
                conviction = min(0.6, abs(best_q) * 0.5 + consensus * 0.3)

            opportunities.append(AdversarialRLOpportunity(
                signal_type=signal_type,
                selected_archetype=self.current_archetype,
                action=best_action,
                q_value=best_q,
                capability_score=self.capability_score,
                adversarial_robustness=self.adversarial_robustness,
                ensemble_consensus=consensus,
                conviction=conviction,
                metadata={'archetype_label': (
                    self.vq_codebook.label_map.get(self.current_archetype or 0, "")
                )}
            ))

        # 决定主信号 (优先级: 能力边界 > 对抗告警 > 集成分歧 > 原型匹配)
        priority_order = [
            AdversarialRLSignalType.CAPABILITY_BOUNDARY,
            AdversarialRLSignalType.ADVERSARIAL_ALARM,
            AdversarialRLSignalType.ENSEMBLE_DISAGREE,
            AdversarialRLSignalType.STRONG_BUY,
            AdversarialRLSignalType.STRONG_SELL,
            AdversarialRLSignalType.ARCHETYPE_MATCH,
        ]

        primary = None
        for sig_type in priority_order:
            for opp in opportunities:
                if opp.signal_type == sig_type:
                    primary = opp
                    break
            if primary:
                break

        if primary is None:
            signal_type = AdversarialRLSignalType.NEUTRAL
            direction = 0
            conviction = 0.0
            action = ActionType.FLAT
        else:
            signal_type = primary.signal_type
            action = primary.action
            # 方向
            if action in [ActionType.LONG_SMALL, ActionType.LONG_MEDIUM,
                          ActionType.LONG_LARGE]:
                direction = 1
            elif action in [ActionType.SHORT_SMALL, ActionType.SHORT_MEDIUM,
                            ActionType.SHORT_LARGE]:
                direction = -1
            else:
                direction = 0
            conviction = primary.conviction
            # 能力/对抗告警时降低 conviction
            if signal_type in [AdversarialRLSignalType.CAPABILITY_BOUNDARY,
                               AdversarialRLSignalType.ADVERSARIAL_ALARM]:
                conviction *= 0.5  # 保守减半

        signal = AdversarialRLSignal(
            signal_type=signal_type,
            direction=direction,
            conviction=conviction,
            selected_archetype=self.current_archetype,
            action=action,
            adversarial_level=self.current_adversarial_level,
            capability_score=self.capability_score,
            ensemble_consensus=self.ensemble_consensus,
            opportunities=opportunities,
        )

        if signal_type != AdversarialRLSignalType.NEUTRAL:
            self.signal_count += 1

        return signal

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'trajectory_size': len(self.trajectory),
            'current_archetype': self.current_archetype,
            'archetype_label': self.vq_codebook.label_map.get(
                self.current_archetype or 0, "") if self.current_archetype else "",
            'current_adversarial_level': self.current_adversarial_level.value,
            'capability_score': self.capability_score,
            'adversarial_robustness': self.adversarial_robustness,
            'ensemble_consensus': self.ensemble_consensus,
            'vae_error_threshold': self.vae.error_threshold,
            'vae_history_size': len(self.vae.recon_error_history),
            'archetype_usage': self.vq_codebook.usage_count.tolist(),
            'td_errors': self.ensemble_q.td_errors,
            'signal_count': self.signal_count,
            'capability_violations': self.capability_violations,
            'adversarial_alarms': self.adversarial_alarms,
        }

    def get_archetype_stats(self) -> List[Dict[str, Any]]:
        """获取所有原型统计"""
        stats = []
        for i in range(self.N_ARCHETYPES):
            rewards = list(self.archetype_rewards[i])
            stats.append({
                'archetype_id': i,
                'label': self.vq_codebook.label_map.get(i, f"archetype_{i}"),
                'usage_count': int(self.vq_codebook.usage_count[i]),
                'avg_reward': sum(rewards) / len(rewards) if rewards else 0.0,
                'recent_reward': rewards[-1] if rewards else 0.0,
            })
        return stats
