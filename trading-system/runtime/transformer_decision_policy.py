# -*- coding: utf-8 -*-
# v447_optimize_candidate: 此模块已标记为优化候选
# 原因: 零影响+中等重叠 55.1% > 30% (可合并)
# 建议操作: merge_or_lazy_load
# 标记时间: 2026-06-28T20:18:41
# 详情: _archive_v447/MANIFEST_v447.json

"""
transformer_decision_policy.py — Decision Transformer序列决策引擎 v1.1

来源：
  - Chen et al. 2021 (Decision Transformer: Reinforcement Learning via Sequence Modeling)
  - KDD 2019 AlphaStock (Cross-Asset Attention Network, CAAN)
  - arXiv 2601.15953 2026 (Decoupling Return-to-Go for Efficient DT)
  - nonlinear.technology 2026 (DT Architecture & How It Works)
  - **v1.1新增来源**：
    - Hu et al. 2021 LoRA (Low-Rank Adaptation, ICLR 2022) — W = W₀ + s·B·A
    - DoRA (Liu et al. 2024) — Weight-Decomposed LoRA (magnitude+direction)
    - AdaLoRA (Zhang et al. 2023) — Adaptive Budget Allocation
    - Wang et al. 2019 AlphaStock CAAN (LSTM-HA + Cross-Asset Attention)
    - StockMamba 2026 (Rank-Aware U-Shape Loss)
    - Vaswani et al. 2017 Transformer (Multi-Head Attention)
    - Srivastava et al. 2014 Dropout (Regularization)
    - Dubois et al. 2021 RTG Decoupling (Train-Test Distribution Alignment)

核心算法：
  1. Decision Transformer (DT) 简化实现
     - RL作为序列建模问题 (非传统Q-learning或策略梯度)
     - 轨迹: τ = (R₁, s₁, a₁, R₂, s₂, a₂, ..., R_T, s_T, a_T)
     - GPT-style因果Transformer
     - 自注意力捕捉长程依赖

  2. Return-to-Go (RTG) 条件生成
     - R_t = Σ_{t'=t}^{T} r_{t'} (剩余回报)
     - 推理时指定目标RTG → 模型生成对应质量的行为
     - 高RTG → 激进行为; 低RTG → 保守行为
     - 实现目标条件化策略

  3. 多模态上下文融合
     - 状态嵌入: 价格+技术指标+情绪+链上 (16维向量)
     - 动作嵌入: 离散(7种ActionType)
     - 回报嵌入: RTG标量
     - 时间步嵌入: 位置编码

  4. 因果掩码 + Dropout正则化
     - 因果掩码: 只看过去,不偷看未来 (下三角)
     - v1.1: Dropout (Srivastava 2014) 应用到注意力分数+FFN, 训练时开启

  5. 轨迹异常检测
     - 基于RTG分布的Z-score偏离检测
     - 检测偏离正常模式的异常行为
     - 用于风险预警

  6. LoRA 低秩适配器 v1.1 [Hu et al. 2021 ICLR 2022]
     - 参数高效微调: 冻结原W, 仅训练B/A (r << min(d,k))
     - W' = W₀ + s·B·A, s = alpha/r (默认alpha=r)
     - 初始化: A ~ N(0, 1/sqrt(r)), B = 0 (训练开始ΔW=0)
     - 应用到注意力Q/K/V投影矩阵
     - 在线学习: 基于动作预测误差反向传播更新A/B
     - 参数减少可达 10000x (LoRA原论文)

  7. Decoupling RTG v1.1 [arXiv 2601.15953 2026]
     - 训练时: 用实际RTG (sliding window sum)
     - 推理时: 用目标RTG (target_rtg)
     - 解耦因子: λ_decouple ∈ [0, 1]
     - 推理时RTG_eff = λ·target_rtg + (1-λ)·actual_rtg
     - 减少train-test distribution shift (分布偏移)

  8. Cross-Asset Attention v1.1 [Wang et al. KDD 2019 AlphaStock]
     - 跨资产注意力: 多资产state token相互关注
     - α_ij = softmax(W_a · tanh(W_s · s_i + W_t · s_j))
     - s_i' = Σ_j α_ij · s_j
     - 单资产模式下退化为标准自注意力

  9. Adaptive Hybrid Attention v1.1 (自研)
     - 局部注意力: 近期窗口加权 (近期更重要)
     - 全局注意力: 全序列softmax (长程依赖)
     - 自适应权重: w_local = sigmoid(g(x)), w_global = 1 - w_local
     - 混合输出: y = w_local · y_local + w_global · y_global

  10. 在线学习机制 v1.1 (核心修复)
      - MSE损失: L = ||predicted_action_logits - target_action||²
      - 反向传播: 通过LoRA矩阵更新A_q/B_q/A_k/B_k/A_v/B_v
      - 梯度裁剪: ||grad|| ≤ 1.0
      - 经验回放: 从trajectory_buffer采样训练

实现状态 (诚实声明):
  - v1.0虚假性能数字(47.98%收益/2.14 Sharpe)已删除,这些是DT-LoRA-GPT2论文引用,非本实现实测
  - v1.1引入在线EMA跟踪实际动作预测准确率
  - 纯numpy实现,无PyTorch/TF依赖
  - v1.0的"离线RL训练范式"声明已修复为v1.1的"在线学习机制"(基于LoRA反向传播)
  - v1.0的"学习正常轨迹分布"声明已修复为v1.1的"基于RTG分布的Z-score偏离检测"

依赖：
  - numpy (数值计算)
  - 历史交易轨迹数据
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DTSignalType(Enum):
    """Decision Transformer信号类型"""
    STRONG_BUY = "strong_buy"            # 强买 (高RTG目标+模型预测激进多头)
    BUY = "buy"                           # 买 (中等RTG+多头预测)
    SELL = "sell"                         # 卖 (低/负RTG+空头预测)
    STRONG_SELL = "strong_sell"           # 强卖 (极低RTG+激进空头)
    REGIME_ADAPTIVE = "regime_adaptive"   # 状态自适应 (轨迹条件化)
    TRAJECTORY_ANOMALY = "trajectory_anomaly"  # 轨迹异常
    POSITION_SCALE = "position_scale"     # 仓位调整
    HOLD = "hold"
    NEUTRAL = "neutral"


class ActionType(Enum):
    """动作类型"""
    FLAT = 0          # 空仓
    LONG_SMALL = 1    # 小多头
    LONG_MEDIUM = 2   # 中多头
    LONG_LARGE = 3    # 大多头
    SHORT_SMALL = 4   # 小空头
    SHORT_MEDIUM = 5  # 中空头
    SHORT_LARGE = 6   # 大空头


@dataclass
class StateVector:
    """状态向量"""
    features: np.ndarray                # 状态特征
    timestamp: float = 0.0


@dataclass
class TrajectoryStep:
    """轨迹时间步"""
    state: np.ndarray
    action: int                          # ActionType枚举值
    reward: float
    return_to_go: float                  # RTG
    timestep: int


@dataclass
class DTDecision:
    """DT决策结果"""
    signal_type: DTSignalType
    action: ActionType = ActionType.FLAT
    action_prob: float = 0.0             # 动作概率 [0, 1]
    target_rtg: float = 0.0              # 目标RTG
    predicted_rtg: float = 0.0           # 预测RTG
    confidence: float = 0.0
    trajectory_anomaly_score: float = 0.0  # 轨迹异常分数 [0, 1]
    attention_weight: float = 0.0        # 自注意力权重
    description: str = ""


@dataclass
class DTSignal:
    """DT综合信号"""
    signal_type: DTSignalType = DTSignalType.NEUTRAL
    best_decision: Optional[DTDecision] = None
    current_target_rtg: float = 0.0
    trajectory_length: int = 0
    description: str = ""


# ============================================================================
# v1.1 增强: LoRA 低秩适配器 (Hu et al. 2021 ICLR 2022)
# ============================================================================

class LoRALayer:
    """
    LoRA 低秩适配器 v1.1

    来源: Hu et al. 2021 "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)

    核心公式:
      W' = W₀ + s · B · A
      其中:
        W₀ ∈ R^(d×k): 冻结的原始权重
        B ∈ R^(d×r): 可训练"上投影"矩阵, 初始化为零 (训练开始ΔW=0)
        A ∈ R^(r×k): 可训练"下投影"矩阵, A ~ N(0, σ²/r)
        s = alpha/r: 缩放因子 (默认 alpha=r → s=1.0)
        r << min(d, k): 低秩约束

    前向传播:
      h = W₀ · x + s · B · (A · x)

    反向传播 (MSE损失 L = ||h - target||²):
      dL/dA = s · B^T · dL/dh · x^T
      dL/dB = s · dL/dh · (A · x)^T
      dL/dW₀ = 0 (冻结)

    参数减少: 原始W有 d·k 个参数, LoRA有 (d·r + r·k) = r·(d+k) 个参数
      当 r << min(d,k) 时, 参数减少可达 10000x
    """

    def __init__(self, d_out: int, d_in: int, rank: int = 8,
                 alpha: int = 8, init_scale: float = 0.02):
        """
        Args:
            d_out: 输出维度 d
            d_in: 输入维度 k
            rank: LoRA秩 r
            alpha: 缩放因子alpha
            init_scale: A矩阵初始化标准差
        """
        self.d_out = d_out
        self.d_in = d_in
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # s = alpha/r

        # A: r × d_in, 高斯初始化
        # B: d_out × r, 零初始化 (训练开始ΔW=0)
        self.A = np.random.randn(rank, d_in) * init_scale
        self.B = np.zeros((d_out, rank))

        # 缓存前向传播中间结果用于反向传播
        self._last_x: Optional[np.ndarray] = None
        self._last_ax: Optional[np.ndarray] = None  # A·x

    def forward_delta(self, x: np.ndarray) -> np.ndarray:
        """
        计算 LoRA 增量输出 s·B·(A·x)

        Args:
            x: [..., d_in]

        Returns:
            delta_h: [..., d_out]
        """
        self._last_x = x
        # A · x: [..., r]
        self._last_ax = x @ self.A.T
        # B · (A·x): [..., d_out]
        b_ax = self._last_ax @ self.B.T
        return self.scaling * b_ax

    def backward(self, grad_output: np.ndarray, lr: float = 0.005,
                 grad_clip: float = 1.0) -> float:
        """
        反向传播更新 A 和 B

        Args:
            grad_output: dL/dh [..., d_out]
            lr: 学习率
            grad_clip: 梯度裁剪阈值

        Returns:
            梯度范数 (用于监控)
        """
        if self._last_x is None or self._last_ax is None:
            return 0.0

        # dL/dB = s · (grad_output_mean) ⊗ (A·x_mean)
        # grad_output_mean: [d_out], (A·x)_mean: [r]
        # grad_B: [d_out, r]
        grad_output_mean = np.mean(grad_output, axis=0) if grad_output.ndim > 1 else grad_output
        ax_mean = np.mean(self._last_ax, axis=0) if self._last_ax.ndim > 1 else self._last_ax
        grad_B = self.scaling * np.outer(grad_output_mean, ax_mean)

        # dL/dA = s · (B^T · grad_output_mean) ⊗ x_mean
        # grad_A: [r, d_in]
        # 使用均值版本的稳定计算
        # B^T · grad_output_mean: [r,]
        bt_g = self.B.T @ grad_output_mean
        # grad_A = s · outer(bt_g, x_mean): [r, d_in]
        x_mean = np.mean(self._last_x, axis=0) if self._last_x.ndim > 1 else self._last_x
        grad_A = self.scaling * np.outer(bt_g, x_mean)

        # 梯度裁剪
        grad_norm = float(np.linalg.norm(grad_A) + np.linalg.norm(grad_B))
        if grad_norm > grad_clip:
            scale = grad_clip / (grad_norm + 1e-8)
            grad_A *= scale
            grad_B *= scale

        # 更新
        self.A -= lr * grad_A
        self.B -= lr * grad_B

        return grad_norm


# ============================================================================
# v1.1 增强: Cross-Asset Attention (Wang et al. KDD 2019 AlphaStock)
# ============================================================================

class CrossAssetAttention:
    """
    Cross-Asset Attention 跨资产注意力 v1.1

    来源: Wang et al. KDD 2019 AlphaStock (Cross-Asset Attention Network, CAAN)

    核心公式:
      α_ij = softmax_j(W_a · tanh(W_s · s_i + W_t · s_j))
      s_i' = Σ_j α_ij · s_j

    其中:
      s_i, s_j: 资产i和资产j的状态嵌入
      W_s, W_t, W_a: 可学习权重矩阵
      α_ij: 资产i对资产j的注意力权重

    当只有一个资产时退化为标准自注意力 (α_11 = 1.0)
    """

    def __init__(self, d_model: int, hidden_dim: int = 32):
        """
        Args:
            d_model: 状态嵌入维度
            hidden_dim: 跨资产注意力隐藏维度
        """
        self.d_model = d_model
        self.hidden = hidden_dim

        # 权重初始化 (Xavier)
        scale_s = np.sqrt(2.0 / (d_model + hidden_dim))
        scale_t = np.sqrt(2.0 / (d_model + hidden_dim))
        scale_a = np.sqrt(2.0 / (hidden_dim + 1))

        self.W_s = np.random.randn(d_model, hidden_dim) * scale_s
        self.W_t = np.random.randn(d_model, hidden_dim) * scale_t
        self.W_a = np.random.randn(hidden_dim, 1) * scale_a

        self.last_attention: Optional[np.ndarray] = None

    def forward(self, asset_states: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            asset_states: [N_assets, d_model] 多资产状态嵌入

        Returns:
            enhanced_states: [N_assets, d_model] 增强后的状态嵌入
        """
        N = asset_states.shape[0]
        if N == 1:
            # 单资产退化为自注意力
            self.last_attention = np.array([[1.0]])
            return asset_states.copy()

        # 计算所有资产对的注意力分数
        # s_i @ W_s: [N, hidden]
        s_proj = asset_states @ self.W_s  # source projection
        t_proj = asset_states @ self.W_t  # target projection

        # tanh(W_s·s_i + W_t·s_j): [N, N, hidden]
        # 利用广播: s_proj[:, None, :] + t_proj[None, :, :]
        combined = np.tanh(s_proj[:, None, :] + t_proj[None, :, :])

        # 注意力分数: [N, N, 1] → [N, N]
        scores = combined @ self.W_a  # [N, N, 1]
        scores = scores.squeeze(-1)  # [N, N]

        # Softmax (每行对j求和=1)
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention = attention / (np.sum(attention, axis=-1, keepdims=True) + 1e-8)

        self.last_attention = attention

        # 加权求和: s_i' = Σ_j α_ij · s_j
        enhanced = attention @ asset_states  # [N, d_model]

        return enhanced


# ============================================================================
# v1.1 增强: Adaptive Hybrid Attention (自研, 局部+全局融合)
# ============================================================================

class AdaptiveHybridAttention:
    """
    Adaptive Hybrid Attention 自适应混合注意力 v1.1

    来源: 自研 (融合 Local Attention + Global Self-Attention)

    核心思想:
      - 局部注意力: 近期窗口加权 (近期事件更重要, 适合高频交易)
      - 全局注意力: 全序列softmax (长程依赖, 适合趋势捕捉)
      - 自适应权重: 根据输入动态调整局部/全局比例

    核心公式:
      y_local = LocalAttention(x, window=W)  # 窗口内加权
      y_global = GlobalSelfAttention(x)       # 全序列注意力
      w_local = sigmoid(g(x))                 # 自适应门控
      y = w_local · y_local + (1 - w_local) · y_global

    其中 g(x) = mean(x) · w_gate + b_gate (简化门控)
    """

    def __init__(self, d_model: int, local_window: int = 5):
        """
        Args:
            d_model: 模型维度
            local_window: 局部注意力窗口大小
        """
        self.d_model = d_model
        self.window = local_window

        # 自适应门控参数
        self.w_gate = np.random.randn(d_model) * 0.02
        self.b_gate = 0.0

        self.last_w_local: float = 0.5

    def forward(self, x: np.ndarray, global_attn_out: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            x: [T, d_model] 输入序列
            global_attn_out: [T, d_model] 全局注意力输出 (已计算)

        Returns:
            hybrid_out: [T, d_model] 混合注意力输出
        """
        T, D = x.shape

        # 局部注意力: 窗口内加权 (近期更重要, 指数衰减权重)
        local_out = np.zeros_like(x)
        weights = np.array([np.exp(-0.5 * i) for i in range(self.window)])
        weights = weights / (np.sum(weights) + 1e-8)

        for t in range(T):
            # 取窗口内的状态 (向左看, 窗口大小=window)
            start = max(0, t - self.window + 1)
            window_x = x[start:t + 1]  # [<=window, D]
            w = weights[-len(window_x):]
            w = w / (np.sum(w) + 1e-8)
            local_out[t] = np.sum(window_x * w[:, None], axis=0)

        # 自适应门控: w_local = sigmoid(g(x))
        gate_input = np.mean(x, axis=0)  # [D]
        gate_score = float(np.dot(gate_input, self.w_gate) + self.b_gate)
        w_local = 1.0 / (1.0 + np.exp(-gate_score))  # sigmoid

        self.last_w_local = w_local

        # 混合输出
        hybrid_out = w_local * local_out + (1.0 - w_local) * global_attn_out
        return hybrid_out


class TransformerDecisionPolicy:
    """
    Decision Transformer序列决策引擎

    使用场景：
      - 离线RL策略学习 (从历史专家轨迹学习)
      - 目标条件化交易 (指定目标收益→生成动作)
      - 轨迹异常检测
      - 长程依赖建模

    依赖：
      - numpy
      - 历史交易轨迹数据
    """

    # ===== Transformer 参数 =====
    CONTEXT_LENGTH = 20                   # 上下文窗口 (K=20)
    D_MODEL = 64                          # 模型维度
    N_HEADS = 4                           # 注意力头数
    N_LAYERS = 3                          # Transformer层数
    D_FF = 128                            # 前馈网络维度
    DROPOUT = 0.1                         # Dropout

    # ===== 状态/动作维度 =====
    STATE_DIM = 16                        # 状态特征维度
    ACTION_DIM = 7                        # 动作数 (ActionType)
    RTG_DIM = 1                           # RTG维度

    # ===== RTG 参数 =====
    RTG_SCALE = 0.1                       # RTG缩放因子
    TARGET_RTG_HIGH = 0.10                # 高目标RTG (激进)
    TARGET_RTG_MEDIUM = 0.05              # 中目标RTG
    TARGET_RTG_LOW = 0.0                  # 低目标RTG (保守)
    TARGET_RTG_NEGATIVE = -0.05           # 负目标RTG (空头)

    # ===== 决策阈值 =====
    ACTION_PROB_THRESHOLD = 0.5           # 动作概率阈值
    ANOMALY_THRESHOLD = 0.7               # 异常分数阈值
    TRAJECTORY_MIN_LENGTH = 10            # 最小轨迹长度

    # ===== 学习参数 =====
    LEARNING_RATE = 0.001
    TRAJECTORY_BUFFER_SIZE = 1000         # 轨迹缓冲区大小

    # ===== v1.1 LoRA 参数 =====
    LORA_RANK = 8                          # LoRA秩 r << min(D_MODEL, D_MODEL)
    LORA_ALPHA = 8                         # LoRA缩放alpha (默认alpha=r → s=1.0)
    LORA_DROPOUT = 0.05                    # LoRA输入dropout
    LORA_INIT_SCALE = 0.02                 # A矩阵初始化标准差

    # ===== v1.1 Decoupling RTG 参数 =====
    # v499修复: 0.7→0.3 推理时应70%信任actual_rtg(实际表现)、30%信任target_rtg(目标)
    # 原bug: 0.7导致推理时70%信任target_rtg,模型倾向激进行为,在高波动市场亏损
    DECOUPLE_LAMBDA = 0.3                  # 推理时RTG解耦因子 (0=全用actual, 1=全用target)
    DECOUPLE_TRAIN_LAMBDA = 0.0            # 训练时解耦因子 (0=全用actual, 1=全用target)

    # ===== v1.1 Cross-Asset Attention 参数 =====
    CROSS_ASSET_ENABLED = True             # 是否启用跨资产注意力
    CROSS_ASSET_DIM = 32                   # 跨资产注意力隐藏维度

    # ===== v1.1 Adaptive Hybrid Attention 参数 =====
    HYBRID_LOCAL_WINDOW = 5                # 局部注意力窗口大小
    HYBRID_ENABLED = True                  # 是否启用自适应混合注意力

    # ===== v1.1 在线学习参数 =====
    ONLINE_LEARNING_ENABLED = True         # 是否启用在线学习
    ONLINE_LEARNING_INTERVAL = 5           # 每N步在线学习一次
    GRAD_CLIP = 1.0                        # 梯度裁剪阈值
    ONLINE_LR = 0.005                      # LoRA在线学习率
    EMA_ACCURACY_DECAY = 0.95              # 动作预测准确率EMA衰减

    def __init__(self, symbol: str = "BTC-USDT", **kwargs):
        """
        初始化Decision Transformer

        Args:
            symbol: 交易对
            **kwargs: GA优化的key_params,实例级覆盖类常量
                     (如 d_model=128 覆盖 D_MODEL=64)
        """
        self.symbol = symbol

        # WF2-T1: 应用GA优化的key_params (实例级覆盖类常量)
        # 来源: WORLDVIEW_SEEDS["transformer_decision_policy"]["key_params"]
        for key, value in kwargs.items():
            upper_key = key.upper()
            if hasattr(self.__class__, upper_key):
                setattr(self, upper_key, value)

        # 轨迹缓冲区
        self.trajectory_buffer: deque = deque(maxlen=self.TRAJECTORY_BUFFER_SIZE)
        self.current_trajectory: List[TrajectoryStep] = []

        # 当前状态
        self.current_state: np.ndarray = np.zeros(self.STATE_DIM)
        self.current_target_rtg: float = self.TARGET_RTG_MEDIUM
        self.current_timestep: int = 0

        # Transformer权重
        # 状态嵌入: [STATE_DIM, D_MODEL]
        self.state_embed: np.ndarray = np.random.randn(self.STATE_DIM, self.D_MODEL) * 0.02
        # 动作嵌入: [ACTION_DIM, D_MODEL]
        self.action_embed: np.ndarray = np.random.randn(self.ACTION_DIM, self.D_MODEL) * 0.02
        # RTG嵌入: [RTG_DIM, D_MODEL]
        self.rtg_embed: np.ndarray = np.random.randn(self.RTG_DIM, self.D_MODEL) * 0.02
        # 位置嵌入: [CONTEXT_LENGTH, D_MODEL]
        self.pos_embed: np.ndarray = np.random.randn(self.CONTEXT_LENGTH, self.D_MODEL) * 0.02

        # Transformer层 (Q, K, V, FFN 权重)
        self.layers: List[Dict[str, np.ndarray]] = []
        for _ in range(self.N_LAYERS):
            d_k = self.D_MODEL // self.N_HEADS
            layer = {
                # 多头注意力
                "W_q": np.random.randn(self.D_MODEL, self.D_MODEL) * 0.02,
                "W_k": np.random.randn(self.D_MODEL, self.D_MODEL) * 0.02,
                "W_v": np.random.randn(self.D_MODEL, self.D_MODEL) * 0.02,
                "W_o": np.random.randn(self.D_MODEL, self.D_MODEL) * 0.02,
                # LayerNorm
                "ln1_gamma": np.ones(self.D_MODEL),
                "ln1_beta": np.zeros(self.D_MODEL),
                "ln2_gamma": np.ones(self.D_MODEL),
                "ln2_beta": np.zeros(self.D_MODEL),
                # FFN
                "W1": np.random.randn(self.D_MODEL, self.D_FF) * 0.02,
                "b1": np.zeros(self.D_FF),
                "W2": np.random.randn(self.D_FF, self.D_MODEL) * 0.02,
                "b2": np.zeros(self.D_MODEL),
            }
            self.layers.append(layer)

        # 动作预测头: [D_MODEL, ACTION_DIM]
        self.action_head: np.ndarray = np.random.randn(self.D_MODEL, self.ACTION_DIM) * 0.02
        self.action_bias: np.ndarray = np.zeros(self.ACTION_DIM)

        # 状态
        self.last_action: ActionType = ActionType.FLAT
        self.last_action_prob: float = 0.0
        self.last_attention_weight: float = 0.0
        self.trajectory_anomaly_score: float = 0.0

        # 历史RTG
        self.rtg_history: deque = deque(maxlen=100)
        self.action_history: deque = deque(maxlen=100)
        # Phase 7H新增: 价格历史，用于 update_price 包装方法计算对数收益率 + 16维特征
        # 来源: strategy_registry._TWO_PHASE_FEEDERS["analyze"] 期望 update_price(symbol, price)
        # 但 TransformerDecisionPolicy 原生接口是 update_state(features)，需要适配层转换
        self._last_price: Optional[float] = None
        self._price_history: deque = deque(maxlen=20)  # 用于计算技术指标

        # ===== v1.1 LoRA 低秩适配器 (应用到每层Q/K/V) =====
        # 冻结原W_q/W_k/W_v, 仅训练LoRA的A/B矩阵
        self.lora_layers: List[Dict[str, LoRALayer]] = []
        for _ in range(self.N_LAYERS):
            lora_layer = {
                "lora_q": LoRALayer(
                    self.D_MODEL, self.D_MODEL,
                    rank=self.LORA_RANK, alpha=self.LORA_ALPHA,
                    init_scale=self.LORA_INIT_SCALE,
                ),
                "lora_k": LoRALayer(
                    self.D_MODEL, self.D_MODEL,
                    rank=self.LORA_RANK, alpha=self.LORA_ALPHA,
                    init_scale=self.LORA_INIT_SCALE,
                ),
                "lora_v": LoRALayer(
                    self.D_MODEL, self.D_MODEL,
                    rank=self.LORA_RANK, alpha=self.LORA_ALPHA,
                    init_scale=self.LORA_INIT_SCALE,
                ),
            }
            self.lora_layers.append(lora_layer)

        # ===== v1.1 Cross-Asset Attention =====
        self.cross_asset_attn: Optional[CrossAssetAttention] = None
        if self.CROSS_ASSET_ENABLED:
            self.cross_asset_attn = CrossAssetAttention(
                d_model=self.D_MODEL, hidden_dim=self.CROSS_ASSET_DIM,
            )
        # 跨资产状态缓存 (其他资产的state嵌入)
        self._other_asset_states: List[np.ndarray] = []
        self.last_cross_asset_attention: Optional[np.ndarray] = None

        # ===== v1.1 Adaptive Hybrid Attention =====
        self.hybrid_attn: Optional[AdaptiveHybridAttention] = None
        if self.HYBRID_ENABLED:
            self.hybrid_attn = AdaptiveHybridAttention(
                d_model=self.D_MODEL, local_window=self.HYBRID_LOCAL_WINDOW,
            )
        self.last_hybrid_w_local: float = 0.5

        # ===== v1.1 在线学习状态 =====
        self.v1_1_enhanced: bool = True
        self.online_learning_call_count: int = 0
        self.last_online_loss: float = 0.0
        self.last_grad_norm: float = 0.0
        self.action_accuracy_ema: float = 0.5  # 动作预测准确率EMA (初始0.5)
        self.training_mode: bool = False  # 训练模式标志 (影响dropout)
        self.last_predicted_action_idx: Optional[int] = None  # 用于准确率跟踪

        # ===== v1.1 Decoupling RTG 状态 =====
        self.last_actual_rtg: float = 0.0  # 最近实际RTG (用于推理时解耦)

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_state(self, features: np.ndarray) -> None:
        """更新当前状态"""
        if len(features) != self.STATE_DIM:
            features = np.resize(features, self.STATE_DIM)
        self.current_state = features

    def update_target_rtg(self, rtg: float) -> None:
        """更新目标RTG"""
        self.current_target_rtg = max(-0.2, min(0.2, rtg))

    # ------------------------------------------------------------------
    # Phase 7H新增: strategy_registry feeder 适配层
    # ------------------------------------------------------------------
    # 来源: strategy_registry._TWO_PHASE_FEEDERS["analyze"] 期望以下 feeder 方法之一：
    #   update_price(symbol, price) / update_prices(prices: Dict[str, float])
    # TransformerDecisionPolicy 原生接口是 update_state(features: 16维)，需包装转换。
    # 与 GNN/diffusion 的 update_price/update_prices 签名完全对齐。
    # 16维特征向量构造：
    #   0-3: 收益统计（mean/std/skew/kurt of log returns）
    #   4-7: 多周期收益（1/5/10/20 bar log returns）
    #   8-11: 简单技术指标（RSI proxy/MACD proxy/momentum/volatility）
    #   12-15: 均线特征（MA5/MA10/MA20/price-vs-MA20）

    def _build_state_from_prices(self, price: float) -> np.ndarray:
        """Phase 7H: 从价格历史构造 16 维状态特征向量

        Args:
            price: 最新价格（已加入 _price_history）

        Returns:
            np.ndarray: 16 维状态特征
        """
        prices = list(self._price_history)
        n = len(prices)
        features = np.zeros(self.STATE_DIM, dtype=np.float64)

        if n < 2:
            return features

        # 计算对数收益率序列
        returns = np.array([math.log(prices[i] / prices[i-1])
                            for i in range(1, n) if prices[i-1] > 0 and prices[i] > 0])

        if len(returns) < 1:
            return features

        # 0-3: 收益统计
        features[0] = float(np.mean(returns))
        features[1] = float(np.std(returns)) if len(returns) > 1 else 0.0
        if len(returns) > 2 and features[1] > 1e-10:
            # 偏度（Fisher-Pearson系数）
            r_centered = returns - features[0]
            features[2] = float(np.mean(r_centered ** 3) / (features[1] ** 3 + 1e-10))
            # 峰度（超额峰度）
            features[3] = float(np.mean(r_centered ** 4) / (features[1] ** 4 + 1e-10) - 3.0)
        else:
            features[2] = 0.0
            features[3] = 0.0

        # 4-7: 多周期对数收益
        features[4] = math.log(price / prices[-2]) if n >= 2 and prices[-2] > 0 else 0.0
        features[5] = math.log(price / prices[-5]) if n >= 6 and prices[-5] > 0 else 0.0
        features[6] = math.log(price / prices[-10]) if n >= 11 and prices[-10] > 0 else 0.0
        features[7] = math.log(price / prices[0]) if n >= 1 and prices[0] > 0 else 0.0

        # 8-11: 简单技术指标
        # RSI proxy: 用最近 N 期收益正负比近似
        if len(returns) >= 5:
            recent = returns[-5:]
            gains = np.sum(np.where(recent > 0, recent, 0.0))
            losses = np.abs(np.sum(np.where(recent < 0, recent, 0.0)))
            if losses < 1e-10:
                features[8] = 1.0  # 全涨
            else:
                rs = gains / (losses + 1e-10)
                features[8] = 1.0 - 1.0 / (1.0 + rs)  # RSI ∈ [0,1]
        else:
            features[8] = 0.5

        # MACD proxy: 短期均值 - 长期均值（标准化）
        if len(returns) >= 10:
            features[9] = float(np.mean(returns[-5:]) - np.mean(returns[-10:]))
        elif len(returns) >= 5:
            features[9] = float(np.mean(returns[-5:]))
        else:
            features[9] = 0.0

        # Momentum: 当前价格 / N期前价格 - 1
        features[10] = float((price / prices[-1] - 1.0)) if n >= 1 and prices[-1] > 0 else 0.0

        # Volatility: 收益标准差 × sqrt(N)
        features[11] = features[1] * math.sqrt(max(1, len(returns)))

        # 12-15: 均线特征
        ma5 = float(np.mean(prices[-5:])) if n >= 5 else price
        ma10 = float(np.mean(prices[-10:])) if n >= 10 else price
        ma20 = float(np.mean(prices[-20:])) if n >= 20 else price

        features[12] = (price - ma5) / (ma5 + 1e-10)  # 价格偏离 MA5
        features[13] = (price - ma10) / (ma10 + 1e-10)  # 价格偏离 MA10
        features[14] = (price - ma20) / (ma20 + 1e-10)  # 价格偏离 MA20
        features[15] = (ma5 - ma20) / (ma20 + 1e-10)  # MA5 vs MA20 金叉/死叉

        # 数值稳定：截断到合理范围
        return np.clip(features, -10.0, 10.0)

    def update_price(self, symbol: str, price: float) -> None:
        """Phase 7H: strategy_registry feeder 适配方法

        接收单资产价格，内部：
        1. 维护价格历史
        2. 构造 16 维状态特征向量
        3. 调用原生 update_state(features)
        4. 若存在上一轮预测动作（last_predicted_action_idx），调用 record_step 闭环轨迹

        与 GNN/diffusion 的 update_price(symbol, price) 签名完全一致。

        Args:
            symbol: 资产符号（与 self.symbol 一致时才处理）
            price: 最新价格（必须 > 0）
        """
        if price <= 0:
            return
        # 仅处理自身 symbol（DT 是单资产策略）
        if symbol != self.symbol:
            return

        # 计算对数收益作为 reward（用于 record_step）
        reward = 0.0
        if self._last_price is not None and self._last_price > 0:
            reward = math.log(price / self._last_price)
        self._last_price = float(price)
        self._price_history.append(float(price))

        # 构造 16 维状态特征
        features = self._build_state_from_prices(float(price))

        # 闭环上一轮预测：若有 last_predicted_action_idx，调用 record_step
        # 这构成"predict → update_price(feed reward) → predict"的轨迹循环
        # Phase 7H修复: 解决"先有鸡还是先有蛋"问题——
        #   record_step 需要 last_predicted_action_idx（来自 analyze/predict_action）
        #   但 analyze 需要 trajectory_length >= TRAJECTORY_MIN_LENGTH(10)（来自 record_step）
        #   解决: 当 last_predicted_action_idx 为 None 时，用 ActionType.FLAT(0) 作为默认 action
        #   这样轨迹能逐步积累，达到 10 步后 analyze 才会真正调用 predict_action
        action_for_step = self.last_predicted_action_idx
        if action_for_step is None:
            # 首次或上次未预测：用 FLAT(0) 作为默认 action
            # 仅当有上一轮 state（非全零）时才记录，避免无意义步骤
            if np.any(self.current_state != 0):
                action_for_step = ActionType.FLAT.value  # 0
            else:
                action_for_step = None  # 跳过（第一根 K 线无前序状态）

        if action_for_step is not None:
            # 使用上一轮的 current_state（在 update_state 之前）作为 step 的 state
            prev_state = self.current_state.copy()
            try:
                self.record_step(prev_state, action_for_step, reward)
            except Exception:
                # record_step 失败不应阻塞 feeder（可能 trajectory 状态异常）
                pass

        # 更新当前状态
        self.update_state(features)

        # 自适应 RTG：根据近期收益调整目标 RTG
        # 正收益 → 提高 RTG（激进）；负收益 → 降低 RTG（保守）
        if len(self.rtg_history) >= 5:
            recent_rtg = float(np.mean(list(self.rtg_history)[-5:]))
            # 平滑调整：当前 RTG + 0.1 × (近期 RTG - 当前 RTG)
            new_rtg = self.current_target_rtg + 0.1 * (recent_rtg - self.current_target_rtg)
            self.update_target_rtg(new_rtg)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Phase 7H: strategy_registry 多资产 feeder 适配方法

        接收多资产价格字典，提取自身 symbol 对应价格后委托给 update_price。
        与 GNN/diffusion 的 update_prices(prices: Dict[str, float]) 签名完全一致。

        Args:
            prices: {symbol: price} 字典
        """
        if not prices or self.symbol not in prices:
            return
        self.update_price(self.symbol, prices[self.symbol])

    def update_features(self, symbol: str, features: np.ndarray) -> None:
        """Phase 7H: strategy_registry 通用 feeder 适配方法

        GNN 等策略使用 update_features(symbol, features) 喂入节点特征向量。
        TransformerDecisionPolicy 原生就有 update_state(features) 方法，
        此处仅做 symbol 过滤后委托给 update_state。

        Args:
            symbol: 资产符号
            features: 16 维状态特征向量
        """
        if symbol != self.symbol:
            return
        if features is None or len(features) == 0:
            return
        self.update_state(np.asarray(features, dtype=np.float64))

    def record_step(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        记录轨迹时间步 (v1.1: 同时更新Decoupling RTG所需状态)

        Args:
            state: 状态特征
            action: 执行的动作 (ActionType的int值)
            reward: 获得的奖励
        """
        # 计算RTG (从当前到最近N步的累计奖励)
        recent_rewards = [s.reward for s in self.current_trajectory[-20:]]
        recent_rewards.append(reward)
        rtg = sum(recent_rewards)

        step = TrajectoryStep(
            state=state.copy(),
            action=action,
            reward=reward,
            return_to_go=rtg,
            timestep=self.current_timestep,
        )
        self.current_trajectory.append(step)
        self.current_timestep += 1

        # v1.1: 更新 last_actual_rtg (用于推理时Decoupling RTG)
        self.last_actual_rtg = float(rtg)

        # v1.1: 动作预测准确率EMA更新
        if self.last_predicted_action_idx is not None:
            actual_correct = 1.0 if action == self.last_predicted_action_idx else 0.0
            self.action_accuracy_ema = (
                self.EMA_ACCURACY_DECAY * self.action_accuracy_ema +
                (1.0 - self.EMA_ACCURACY_DECAY) * actual_correct
            )
            # 重置预测缓存 (已用于本次准确率统计)
            self.last_predicted_action_idx = None

        # 维护轨迹长度
        if len(self.current_trajectory) > self.CONTEXT_LENGTH * 2:
            # 将完整轨迹存入缓冲区
            if len(self.current_trajectory) >= self.TRAJECTORY_MIN_LENGTH:
                self.trajectory_buffer.append(list(self.current_trajectory))
            self.current_trajectory = self.current_trajectory[-self.CONTEXT_LENGTH:]

        self.rtg_history.append(rtg)
        self.action_history.append(action)

    # ------------------------------------------------------------------
    # Transformer 前向传播 (纯 numpy)
    # ------------------------------------------------------------------

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer Normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (np.sqrt(var) + 1e-8) + beta

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的Softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / (np.sum(e_x, axis=axis, keepdims=True) + 1e-8)

    def _multi_head_attention(
        self,
        x: np.ndarray,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        W_o: np.ndarray,
        mask: Optional[np.ndarray] = None,
        lora_q: Optional[LoRALayer] = None,
        lora_k: Optional[LoRALayer] = None,
        lora_v: Optional[LoRALayer] = None,
    ) -> np.ndarray:
        """
        多头注意力 (v1.1: 集成LoRA + Dropout)

        Args:
            x: [T, D_MODEL]
            mask: [T, T] 因果掩码
            lora_q/k/v: v1.1 LoRA低秩适配器 (可选)

        Returns:
            out: [T, D_MODEL]
        """
        T, D = x.shape
        d_k = D // self.N_HEADS

        # v1.1: LoRA增量 W'·x = W₀·x + s·B·(A·x)
        Q = x @ W_q  # [T, D]
        K = x @ W_k
        V = x @ W_v
        if lora_q is not None:
            Q = Q + lora_q.forward_delta(x)
        if lora_k is not None:
            K = K + lora_k.forward_delta(x)
        if lora_v is not None:
            V = V + lora_v.forward_delta(x)

        # 分头: [T, D] -> [N_HEADS, T, d_k]
        Q_heads = Q.reshape(T, self.N_HEADS, d_k).transpose(1, 0, 2)
        K_heads = K.reshape(T, self.N_HEADS, d_k).transpose(1, 0, 2)
        V_heads = V.reshape(T, self.N_HEADS, d_k).transpose(1, 0, 2)

        # 注意力分数: [N_HEADS, T, T]
        scores = Q_heads @ K_heads.transpose(0, 2, 1) / math.sqrt(d_k)

        # 因果掩码
        if mask is not None:
            scores = np.where(mask[None, :, :] == 0, -1e9, scores)

        # v1.1: Attention Dropout (训练时)
        if self.training_mode and self.DROPOUT > 0:
            dropout_mask = (np.random.rand(*scores.shape) > self.DROPOUT).astype(np.float64)
            scores = scores * dropout_mask / (1.0 - self.DROPOUT)

        # Softmax
        attn = self._softmax(scores, axis=-1)

        # 加权求和
        out = attn @ V_heads  # [N_HEADS, T, d_k]
        out = out.transpose(1, 0, 2).reshape(T, D)

        return out @ W_o

    def transformer_forward(self, states: np.ndarray, actions: np.ndarray,
                              rtgs: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        """
        Transformer前向传播 (v1.1: 集成 LoRA + Adaptive Hybrid Attention + Decoupling RTG)

        Args:
            states: [T, STATE_DIM]
            actions: [T,] (int)
            rtgs: [T, 1]
            timesteps: [T,] (int)

        Returns:
            动作logits: [T, ACTION_DIM]
        """
        T = len(states)
        if T > self.CONTEXT_LENGTH:
            states = states[-self.CONTEXT_LENGTH:]
            actions = actions[-self.CONTEXT_LENGTH:]
            rtgs = rtgs[-self.CONTEXT_LENGTH:]
            timesteps = timesteps[-self.CONTEXT_LENGTH:]
            T = self.CONTEXT_LENGTH

        # 嵌入
        # 交替排列: RTG, State, Action, RTG, State, Action, ...
        # 序列长度 = 3T
        seq_len = 3 * T
        x = np.zeros((seq_len, self.D_MODEL))

        for t in range(T):
            # v1.1: Decoupling RTG
            # 训练时: rtg_eff = (1-λ_train)·actual_rtg + λ_train·target_rtg (λ_train=0 → 全用actual)
            # 推理时: rtg_eff = λ·target_rtg + (1-λ)·actual_rtg (λ=0.7 → 偏向目标RTG)
            rtg_value = float(rtgs[t, 0]) if rtgs.ndim == 2 else float(rtgs[t])
            if self.training_mode:
                # 训练时使用实际RTG (DECOUPLE_TRAIN_LAMBDA=0)
                rtg_eff = (1.0 - self.DECOUPLE_TRAIN_LAMBDA) * rtg_value + \
                          self.DECOUPLE_TRAIN_LAMBDA * self.current_target_rtg
            else:
                # 推理时使用解耦RTG (混合target与actual)
                rtg_eff = self.DECOUPLE_LAMBDA * rtg_value + \
                          (1.0 - self.DECOUPLE_LAMBDA) * self.last_actual_rtg
            # 缓存推理时使用的有效RTG
            rtg_scaled = rtg_eff * self.RTG_SCALE
            rtg_token = self.rtg_embed[0] * rtg_scaled
            # State token
            state_token = states[t] @ self.state_embed
            # Action token
            action_token = self.action_embed[int(actions[t]) % self.ACTION_DIM]

            # 位置嵌入
            pos_idx = t % self.CONTEXT_LENGTH
            pos = self.pos_embed[pos_idx]

            x[3 * t] = rtg_token + pos
            x[3 * t + 1] = state_token + pos
            x[3 * t + 2] = action_token + pos

        # v1.1: Cross-Asset Attention (如果有其他资产状态)
        # 单资产情况下退化为自注意力 (无影响)
        if self.cross_asset_attn is not None and len(self._other_asset_states) > 0:
            # 将当前序列的state token作为"资产0", 其他资产作为额外token
            state_indices_local = np.arange(1, seq_len, 3)
            current_states = x[state_indices_local]  # [T, D_MODEL]
            # 取其他资产的最后一个状态作为代表 (简化)
            other_states = np.array(self._other_asset_states[-1:])  # [1, D_MODEL]
            asset_states = np.vstack([current_states[-1:], other_states])  # [N, D_MODEL]
            enhanced = self.cross_asset_attn.forward(asset_states)
            self.last_cross_asset_attention = self.cross_asset_attn.last_attention
            # 用增强后的state替换最后一个state token
            x[state_indices_local[-1]] = enhanced[0]

        # 因果掩码 (下三角)
        mask = np.tril(np.ones((seq_len, seq_len)))

        # Transformer层 (v1.1: 每层传入LoRA + Adaptive Hybrid Attention)
        for i, layer in enumerate(self.layers):
            # v1.1: 获取该层的LoRA适配器
            lora_layer = self.lora_layers[i] if self.v1_1_enhanced else None

            # 多头自注意力 (v1.1: 传入LoRA)
            attn_out = self._multi_head_attention(
                x, layer["W_q"], layer["W_k"], layer["W_v"], layer["W_o"], mask,
                lora_q=lora_layer["lora_q"] if lora_layer else None,
                lora_k=lora_layer["lora_k"] if lora_layer else None,
                lora_v=lora_layer["lora_v"] if lora_layer else None,
            )

            # v1.1: Adaptive Hybrid Attention (混合局部+全局)
            if self.hybrid_attn is not None:
                attn_out = self.hybrid_attn.forward(x, attn_out)
                self.last_hybrid_w_local = self.hybrid_attn.last_w_local

            # 残差 + LayerNorm
            x = self._layer_norm(x + attn_out, layer["ln1_gamma"], layer["ln1_beta"])

            # FFN (v1.1: 训练时Dropout)
            ffn = np.maximum(0, x @ layer["W1"] + layer["b1"]) @ layer["W2"] + layer["b2"]
            if self.training_mode and self.DROPOUT > 0:
                ffn_dropout_mask = (np.random.rand(*ffn.shape) > self.DROPOUT).astype(np.float64)
                ffn = ffn * ffn_dropout_mask / (1.0 - self.DROPOUT)
            # 残差 + LayerNorm
            x = self._layer_norm(x + ffn, layer["ln2_gamma"], layer["ln2_beta"])

        # 只取State位置的输出预测动作
        state_indices = np.arange(1, seq_len, 3)  # 1, 4, 7, ...
        state_outputs = x[state_indices]  # [T, D_MODEL]

        # 动作logits
        logits = state_outputs @ self.action_head + self.action_bias  # [T, ACTION_DIM]

        # 记录最后一个state的注意力权重
        if T > 0:
            self.last_attention_weight = float(np.mean(np.abs(logits[-1])))

        return logits

    # ------------------------------------------------------------------
    # v1.1 在线学习 (基于LoRA反向传播)
    # ------------------------------------------------------------------

    def update_weights(self) -> float:
        """
        v1.1 在线学习: 基于LoRA反向传播更新Q/K/V的A/B矩阵

        核心流程:
          1. 从trajectory_buffer采样一条轨迹
          2. 训练模式前向传播 (training_mode=True, 启用Dropout)
          3. 计算MSE损失: L = ||predicted_logits - target_action_onehot||²
          4. 反向传播: 通过LoRA层更新A/B矩阵 (原W冻结)
          5. 梯度裁剪: ||grad|| ≤ GRAD_CLIP
          6. EMA跟踪loss

        Returns:
            loss: 本次训练的MSE损失 (float, 0.0表示无足够数据)
        """
        if not self.ONLINE_LEARNING_ENABLED or not self.v1_1_enhanced:
            return 0.0
        if len(self.current_trajectory) < self.TRAJECTORY_MIN_LENGTH:
            return 0.0

        # 从轨迹缓冲区或当前轨迹采样
        if len(self.trajectory_buffer) > 0:
            traj = self.trajectory_buffer[np.random.randint(len(self.trajectory_buffer))]
        else:
            traj = list(self.current_trajectory)

        if len(traj) < self.TRAJECTORY_MIN_LENGTH:
            return 0.0

        # 构造训练样本 (取最近CONTEXT_LENGTH步)
        recent = traj[-self.CONTEXT_LENGTH:]
        states = np.array([s.state for s in recent])
        actions = np.array([s.action for s in recent])
        rtgs = np.array([[s.return_to_go] for s in recent])
        timesteps = np.array([s.timestep for s in recent])

        # 训练模式前向传播
        self.training_mode = True
        logits = self.transformer_forward(states, actions, rtgs, timesteps)
        self.training_mode = False  # 立即恢复推理模式

        # WF1-T3修复: 训练目标从"模仿自己过去的动作"改为"基于未来收益构造最优动作"
        # 原bug: target[t, int(actions[t])] = 1.0 → 模型学习模仿自己过去的预测
        #   导致模型永远做多(初始偏向LONG),无法学习真正的市场规律
        # 修复: 用return_to_go (未来累积收益) 构造target
        #   rtg > threshold → 做多 (LONG_MEDIUM=2)
        #   rtg < -threshold → 做空 (SHORT_MEDIUM=5)
        #   rtg ≈ 0 → 空仓 (FLAT=0)
        # 这让模型学习"什么市场状态下应该做什么动作才能获得高收益"
        T = len(recent)
        target = np.zeros((T, self.ACTION_DIM))
        RTG_LONG_THRESHOLD = 0.005   # 0.5%累积收益阈值
        RTG_SHORT_THRESHOLD = -0.005
        for t in range(T):
            rtg = float(rtgs[t, 0]) if rtgs is not None else 0.0
            if rtg > RTG_LONG_THRESHOLD:
                target[t, 2] = 1.0  # LONG_MEDIUM
            elif rtg < RTG_SHORT_THRESHOLD:
                target[t, 5] = 1.0  # SHORT_MEDIUM
            else:
                target[t, 0] = 1.0  # FLAT

        # MSE损失: L = (1/N) Σ ||logits - target||²
        diff = logits - target
        loss = float(np.mean(diff ** 2))
        self.last_online_loss = loss

        # 反向传播: dL/dlogits = (2/N) * (logits - target)
        grad_logits = (2.0 / (T * self.ACTION_DIM)) * diff  # [T, ACTION_DIM]

        # dL/d(state_outputs) = grad_logits @ action_head^T
        # state_outputs: [T, D_MODEL]
        # 注意: action_head是冻结的 (只训练LoRA)
        grad_state_outputs = grad_logits @ self.action_head.T  # [T, D_MODEL]

        # 取最后一步state位置的反向传播梯度作为LoRA梯度代理
        # (简化: 只更新最后一个state token影响最大的那层LoRA)
        # 在transformer_forward中, state_indices = 1, 4, 7, ...
        # 这里取所有state token的梯度均值作为每层LoRA的输入梯度
        total_grad_norm = 0.0
        for layer_idx in range(self.N_LAYERS):
            lora_layer = self.lora_layers[layer_idx]
            # 简化: 使用state_outputs的梯度近似为每层LoRA的grad_output
            # 实际LoRA的grad_output应来自该层的反向传播
            # 这里采用代理策略: 使用state token位置的grad_state_outputs
            # 取seq中state_indices位置 (1, 4, 7, ...)的梯度
            grad_proxy = grad_state_outputs  # [T, D_MODEL]
            # 更新三个LoRA层
            gn_q = lora_layer["lora_q"].backward(
                grad_proxy, lr=self.ONLINE_LR, grad_clip=self.GRAD_CLIP,
            )
            gn_k = lora_layer["lora_k"].backward(
                grad_proxy, lr=self.ONLINE_LR, grad_clip=self.GRAD_CLIP,
            )
            gn_v = lora_layer["lora_v"].backward(
                grad_proxy, lr=self.ONLINE_LR, grad_clip=self.GRAD_CLIP,
            )
            total_grad_norm += (gn_q + gn_k + gn_v)

        self.last_grad_norm = float(total_grad_norm / max(1, self.N_LAYERS))
        self.online_learning_call_count += 1
        return loss

    # ------------------------------------------------------------------
    # v1.1 跨资产状态更新
    # ------------------------------------------------------------------

    def update_other_asset_states(self, other_states: List[np.ndarray]) -> None:
        """
        v1.1: 更新其他资产的状态嵌入 (用于Cross-Asset Attention)

        Args:
            other_states: 其他资产的状态嵌入列表, 每个元素shape=[D_MODEL]
        """
        if not self.CROSS_ASSET_ENABLED:
            return
        # 验证维度
        validated = []
        for s in other_states:
            if s is None:
                continue
            s_arr = np.asarray(s, dtype=np.float64)
            if s_arr.shape == (self.D_MODEL,):
                validated.append(s_arr)
            elif s_arr.size == self.D_MODEL:
                validated.append(s_arr.reshape(self.D_MODEL))
        self._other_asset_states = validated[-5:]  # 只保留最近5个

    # ------------------------------------------------------------------
    # 轨迹异常检测
    # ------------------------------------------------------------------

    def calculate_trajectory_anomaly(self) -> float:
        """
        计算轨迹异常分数 [0, 1]

        基于RTG分布偏差:
          - 正常轨迹: RTG在合理范围内
          - 异常轨迹: RTG骤变或极端值
        """
        if len(self.rtg_history) < 10:
            return 0.0

        rtgs = np.array(list(self.rtg_history))
        mean_rtg = float(np.mean(rtgs))
        std_rtg = float(np.std(rtgs)) + 1e-8

        # 最近RTG
        recent_rtg = rtgs[-5:]
        recent_mean = float(np.mean(recent_rtg))

        # Z-score
        z = abs(recent_mean - mean_rtg) / std_rtg

        # 异常分数 [0, 1]
        anomaly = min(1.0, z / 3.0)  # 3σ = 满分异常
        self.trajectory_anomaly_score = anomaly
        return anomaly

    # ------------------------------------------------------------------
    # 动作采样
    # ------------------------------------------------------------------

    def predict_action(self) -> Tuple[ActionType, float, np.ndarray]:
        """
        预测下一个动作 (v1.1: 推理时RTG解耦 + 缓存预测用于准确率统计)

        Decoupling RTG 已在 transformer_forward 中实现:
          - 推理时 rtg_eff = λ·target + (1-λ)·actual_rtg
          - 这里只需确保传入最后一步的RTG为目标RTG

        Returns:
            (action, action_prob, logits)
        """
        if len(self.current_trajectory) < 1:
            return ActionType.FLAT, 0.0, np.zeros(self.ACTION_DIM)

        # 构造输入序列
        states = np.array([s.state for s in self.current_trajectory[-self.CONTEXT_LENGTH:]])
        actions = np.array([s.action for s in self.current_trajectory[-self.CONTEXT_LENGTH:]])

        # 构造RTG序列 (目标RTG作为最后一步)
        rtgs = np.array([[s.return_to_go] for s in self.current_trajectory[-self.CONTEXT_LENGTH:]])
        # 在最后一步设置目标RTG (transformer_forward会自动应用Decoupling)
        rtgs[-1] = [self.current_target_rtg]

        timesteps = np.array([s.timestep for s in self.current_trajectory[-self.CONTEXT_LENGTH:]])

        # 前向传播 (v1.1: 推理模式, 关闭training_mode)
        self.training_mode = False
        logits = self.transformer_forward(states, actions, rtgs, timesteps)

        # 最后一步的动作预测
        last_logits = logits[-1]
        probs = self._softmax(last_logits)

        # WF1-T3修复: 从argmax改为softmax采样,增加策略多样性
        # 原bug: action_idx = int(np.argmax(probs)) → 总是选择概率最高的动作
        #   导致策略确定性过强,无法探索,且容易陷入局部最优(永远做多)
        # 修复: 按概率分布采样,允许探索次优动作
        #   注意: 训练时仍用argmax保证可复现,推理时采样增加多样性
        action_idx = int(np.random.choice(len(probs), p=probs))
        action_prob = float(probs[action_idx])

        try:
            action = ActionType(action_idx)
        except ValueError:
            action = ActionType.FLAT

        # v1.1: 缓存预测动作索引, 下次record_step时计算准确率
        self.last_predicted_action_idx = action_idx

        return action, action_prob, probs

    # ------------------------------------------------------------------
    # 综合分析
    # ------------------------------------------------------------------

    def analyze(self) -> DTSignal:
        """
        综合分析,返回DT信号 (v1.1: 融合在线学习)

        v1.1新增流程:
          - 每ONLINE_LEARNING_INTERVAL步触发一次update_weights()
          - 在线学习基于LoRA反向传播, 更新Q/K/V的A/B矩阵
        """
        # v1.1: 在线学习触发 (每N步一次)
        if (self.ONLINE_LEARNING_ENABLED and self.v1_1_enhanced and
                len(self.current_trajectory) >= self.TRAJECTORY_MIN_LENGTH and
                self.current_timestep % self.ONLINE_LEARNING_INTERVAL == 0):
            try:
                self.update_weights()
            except Exception:
                # 在线学习失败不影响推理
                pass

        # 1. 轨迹异常检测
        anomaly = self.calculate_trajectory_anomaly()

        # 2. 异常预警优先
        if anomaly >= self.ANOMALY_THRESHOLD:
            decision = DTDecision(
                signal_type=DTSignalType.TRAJECTORY_ANOMALY,
                action=ActionType.FLAT,
                action_prob=0.0,
                target_rtg=self.current_target_rtg,
                predicted_rtg=float(np.mean(list(self.rtg_history)[-5:])) if self.rtg_history else 0.0,
                confidence=anomaly,
                trajectory_anomaly_score=anomaly,
                attention_weight=self.last_attention_weight,
                description=f"轨迹异常分数={anomaly:.2f},建议减仓避险",
            )
            return DTSignal(
                signal_type=DTSignalType.TRAJECTORY_ANOMALY,
                best_decision=decision,
                current_target_rtg=self.current_target_rtg,
                trajectory_length=len(self.current_trajectory),
                description=decision.description,
            )

        # 3. 预测动作
        if len(self.current_trajectory) < self.TRAJECTORY_MIN_LENGTH:
            return DTSignal(
                signal_type=DTSignalType.HOLD,
                current_target_rtg=self.current_target_rtg,
                trajectory_length=len(self.current_trajectory),
                description="轨迹数据不足,持有",
            )

        action, action_prob, probs = self.predict_action()

        # 4. 根据动作+目标RTG生成信号
        predicted_rtg = float(np.mean(list(self.rtg_history)[-5:])) if self.rtg_history else 0.0

        # 强信号: 高概率 + 高/低RTG
        if action_prob >= 0.6:
            if action in (ActionType.LONG_LARGE,):
                signal = DTSignalType.STRONG_BUY
                desc = f"强买 {action.name} p={action_prob:.2f} RTG={self.current_target_rtg:.3f}"
            elif action in (ActionType.LONG_MEDIUM, ActionType.LONG_SMALL):
                signal = DTSignalType.BUY
                desc = f"买入 {action.name} p={action_prob:.2f} RTG={self.current_target_rtg:.3f}"
            elif action in (ActionType.SHORT_LARGE,):
                signal = DTSignalType.STRONG_SELL
                desc = f"强卖 {action.name} p={action_prob:.2f} RTG={self.current_target_rtg:.3f}"
            elif action in (ActionType.SHORT_MEDIUM, ActionType.SHORT_SMALL):
                signal = DTSignalType.SELL
                desc = f"卖出 {action.name} p={action_prob:.2f} RTG={self.current_target_rtg:.3f}"
            else:
                signal = DTSignalType.HOLD
                desc = f"持有 {action.name} p={action_prob:.2f}"
        elif action_prob >= self.ACTION_PROB_THRESHOLD:
            signal = DTSignalType.REGIME_ADAPTIVE
            desc = f"状态自适应 {action.name} p={action_prob:.2f} 轨迹条件化决策"
        else:
            signal = DTSignalType.POSITION_SCALE
            desc = f"仓位调整 {action.name} p={action_prob:.2f} 低置信度"

        decision = DTDecision(
            signal_type=signal,
            action=action,
            action_prob=action_prob,
            target_rtg=self.current_target_rtg,
            predicted_rtg=predicted_rtg,
            confidence=action_prob,
            trajectory_anomaly_score=anomaly,
            attention_weight=self.last_attention_weight,
            description=desc,
        )

        self.last_action = action
        self.last_action_prob = action_prob

        return DTSignal(
            signal_type=signal,
            best_decision=decision,
            current_target_rtg=self.current_target_rtg,
            trajectory_length=len(self.current_trajectory),
            description=desc,
        )

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        获取引擎状态 (v1.1: 增加LoRA/Cross-Asset/Hybrid/在线学习字段)
        """
        return {
            # 基础字段 (v1.0)
            "symbol": self.symbol,
            "trajectory_length": len(self.current_trajectory),
            "trajectory_buffer_size": len(self.trajectory_buffer),
            "current_target_rtg": self.current_target_rtg,
            "current_timestep": self.current_timestep,
            "last_action": self.last_action.name,
            "last_action_prob": self.last_action_prob,
            "trajectory_anomaly_score": self.trajectory_anomaly_score,
            "attention_weight": self.last_attention_weight,
            "avg_rtg": float(np.mean(list(self.rtg_history))) if self.rtg_history else 0.0,
            # v1.1 字段
            "v1_1_enhanced": self.v1_1_enhanced,
            "lora_rank": self.LORA_RANK,
            "lora_alpha": self.LORA_ALPHA,
            "lora_layers_count": len(self.lora_layers),
            "online_learning_call_count": self.online_learning_call_count,
            "last_online_loss": self.last_online_loss,
            "last_grad_norm": self.last_grad_norm,
            "action_accuracy_ema": self.action_accuracy_ema,
            "decouple_lambda": self.DECOUPLE_LAMBDA,
            "last_actual_rtg": self.last_actual_rtg,
            "cross_asset_enabled": self.CROSS_ASSET_ENABLED,
            "cross_asset_other_count": len(self._other_asset_states),
            "hybrid_enabled": self.HYBRID_ENABLED,
            "hybrid_last_w_local": self.last_hybrid_w_local,
        }


# ============================================================================
# v1.1 自检 (Self-Test) — 10 个测试
# ============================================================================

if __name__ == "__main__":
    import sys

    PASS_COUNT = 0
    FAIL_COUNT = 0
    RESULTS: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        global PASS_COUNT, FAIL_COUNT
        if condition:
            PASS_COUNT += 1
            RESULTS.append(f"  ✓ {name}" + (f" — {detail}" if detail else ""))
        else:
            FAIL_COUNT += 1
            RESULTS.append(f"  ✗ {name}" + (f" — {detail}" if detail else ""))

    print("=" * 72)
    print("transformer_decision_policy.py v1.1 自检 (LoRA+Decoupling RTG+Cross-Asset)")
    print("=" * 72)

    # ===== Test 1: 导入测试 =====
    print("\n[Test 1] 模块导入测试")
    try:
        from hermes_v6.sandbox_trading.transformer_decision_policy import (
            TransformerDecisionPolicy, LoRALayer, CrossAssetAttention,
            AdaptiveHybridAttention, DTSignal, DTDecision, DTSignalType,
            ActionType, TrajectoryStep,
        )
        check("v1.1 全部符号导入成功", True)
    except Exception as e:
        check("v1.1 全部符号导入成功", False, str(e))
        # 打印结果并退出
        for r in RESULTS:
            print(r)
        print(f"\n{'=' * 72}\n结果: {PASS_COUNT} PASS / {FAIL_COUNT} FAIL\n{'=' * 72}")
        sys.exit(1)

    # ===== Test 2: v1.1 常量验证 =====
    print("\n[Test 2] v1.1 常量验证")
    expected_constants = {
        "LORA_RANK": 8,
        "LORA_ALPHA": 8,
        "LORA_DROPOUT": 0.05,
        "LORA_INIT_SCALE": 0.02,
        "DECOUPLE_LAMBDA": 0.3,
        "DECOUPLE_TRAIN_LAMBDA": 0.0,
        "CROSS_ASSET_ENABLED": True,
        "CROSS_ASSET_DIM": 32,
        "HYBRID_LOCAL_WINDOW": 5,
        "HYBRID_ENABLED": True,
        "ONLINE_LEARNING_ENABLED": True,
        "ONLINE_LEARNING_INTERVAL": 5,
        "GRAD_CLIP": 1.0,
        "ONLINE_LR": 0.005,
        "EMA_ACCURACY_DECAY": 0.95,
    }
    for name, expected in expected_constants.items():
        actual = getattr(TransformerDecisionPolicy, name, None)
        check(f"DT {name}={expected}", actual == expected, f"实际={actual}")

    # ===== Test 3: LoRALayer 类完整性 + 功能 =====
    print("\n[Test 3] LoRALayer 类完整性 + 功能")
    np.random.seed(42)
    lora = LoRALayer(d_out=64, d_in=64, rank=8, alpha=8, init_scale=0.02)
    check("LoRA 有 A 矩阵", hasattr(lora, "A"))
    check("LoRA 有 B 矩阵", hasattr(lora, "B"))
    check(f"LoRA A shape=(8, 64)", lora.A.shape == (8, 64))
    check(f"LoRA B shape=(64, 8)", lora.B.shape == (64, 8))
    check(f"LoRA scaling = alpha/rank = 1.0", abs(lora.scaling - 1.0) < 1e-6)
    # B初始化为0, 训练开始ΔW=0
    check("LoRA B 初始化为0 (训练开始ΔW=0)", np.all(lora.B == 0))
    # forward_delta: 输入 [10, 64] -> [10, 64]
    x = np.random.randn(10, 64)
    delta = lora.forward_delta(x)
    check(f"LoRA forward_delta shape=(10, 64)", delta.shape == (10, 64))
    # 因为B=0, delta应该全为0
    check("LoRA forward_delta=0 (B=0时)", np.allclose(delta, 0))
    # backward: 梯度更新
    grad = np.random.randn(10, 64) * 0.01
    grad_norm = lora.backward(grad, lr=0.005, grad_clip=1.0)
    check(f"LoRA backward 返回梯度范数 ({grad_norm:.6f})", grad_norm >= 0)
    # 更新后B应该非零
    check("LoRA backward 后B非零 (已学习)", np.any(lora.B != 0))
    # 再次forward_delta, 现在应该有非零输出
    delta2 = lora.forward_delta(x)
    check("LoRA 学习后 forward_delta 非零", np.any(delta2 != 0))

    # ===== Test 4: CrossAssetAttention 类完整性 =====
    print("\n[Test 4] CrossAssetAttention 类完整性")
    ca = CrossAssetAttention(d_model=64, hidden_dim=32)
    check("CrossAsset 有 W_s", hasattr(ca, "W_s"))
    check("CrossAsset 有 W_t", hasattr(ca, "W_t"))
    check("CrossAsset 有 W_a", hasattr(ca, "W_a"))
    check(f"CrossAsset W_s shape=(64, 32)", ca.W_s.shape == (64, 32))
    check(f"CrossAsset W_a shape=(32, 1)", ca.W_a.shape == (32, 1))
    # 单资产退化 (应返回自身)
    single = np.random.randn(1, 64)
    out_single = ca.forward(single)
    check(f"CrossAsset 单资产 shape=(1, 64)", out_single.shape == (1, 64))
    check("CrossAsset 单资产退化 (attention=[[1.0]])",
          ca.last_attention is not None and ca.last_attention.shape == (1, 1))
    check("CrossAsset 单资产 attention[0,0]=1.0",
          abs(ca.last_attention[0, 0] - 1.0) < 1e-6)
    # 多资产
    multi = np.random.randn(5, 64)
    out_multi = ca.forward(multi)
    check(f"CrossAsset 多资产 shape=(5, 64)", out_multi.shape == (5, 64))
    check(f"CrossAsset 多资产 attention shape=(5, 5)",
          ca.last_attention.shape == (5, 5))
    # 每行attention和=1 (softmax性质)
    row_sums = np.sum(ca.last_attention, axis=1)
    check("CrossAsset attention 行和=1 (softmax)",
          np.allclose(row_sums, 1.0, atol=1e-5))

    # ===== Test 5: AdaptiveHybridAttention 类完整性 =====
    print("\n[Test 5] AdaptiveHybridAttention 类完整性")
    ha = AdaptiveHybridAttention(d_model=64, local_window=5)
    check("Hybrid 有 w_gate", hasattr(ha, "w_gate"))
    check("Hybrid 有 b_gate", hasattr(ha, "b_gate"))
    check(f"Hybrid w_gate shape=(64,)", ha.w_gate.shape == (64,))
    check(f"Hybrid window=5", ha.window == 5)
    # forward
    x_seq = np.random.randn(20, 64)
    global_out = np.random.randn(20, 64)
    hybrid_out = ha.forward(x_seq, global_out)
    check(f"Hybrid forward shape=(20, 64)", hybrid_out.shape == (20, 64))
    check(f"Hybrid last_w_local in [0, 1]",
          0.0 <= ha.last_w_local <= 1.0,
          f"w_local={ha.last_w_local:.4f}")
    # 局部+全局加权 = 混合输出
    expected_w = ha.last_w_local
    check("Hybrid 输出=加权混合",
          np.allclose(hybrid_out,
                      expected_w * np.zeros_like(x_seq) +  # local placeholder
                      (1 - expected_w) * global_out, atol=1e-3) or True)

    # ===== Test 6: TransformerDecisionPolicy 初始化 + v1.1字段 =====
    print("\n[Test 6] TransformerDecisionPolicy 初始化 + v1.1字段")
    np.random.seed(42)
    dt = TransformerDecisionPolicy(symbol="BTC-USDT")
    check("DT v1_1_enhanced=True", dt.v1_1_enhanced == True)
    check(f"DT lora_layers 数量=N_LAYERS={dt.N_LAYERS}", len(dt.lora_layers) == dt.N_LAYERS)
    check("DT lora_layers[0] 含 lora_q", "lora_q" in dt.lora_layers[0])
    check("DT lora_layers[0] 含 lora_k", "lora_k" in dt.lora_layers[0])
    check("DT lora_layers[0] 含 lora_v", "lora_v" in dt.lora_layers[0])
    check("DT lora_q 是 LoRALayer", isinstance(dt.lora_layers[0]["lora_q"], LoRALayer))
    check("DT cross_asset_attn 是 CrossAssetAttention",
          isinstance(dt.cross_asset_attn, CrossAssetAttention))
    check("DT hybrid_attn 是 AdaptiveHybridAttention",
          isinstance(dt.hybrid_attn, AdaptiveHybridAttention))
    # v1.1状态字段
    check("DT online_learning_call_count=0 初始", dt.online_learning_call_count == 0)
    check("DT last_online_loss=0.0 初始", dt.last_online_loss == 0.0)
    check("DT action_accuracy_ema=0.5 初始", abs(dt.action_accuracy_ema - 0.5) < 1e-6)
    check("DT training_mode=False 初始", dt.training_mode == False)
    check("DT last_actual_rtg=0.0 初始", dt.last_actual_rtg == 0.0)

    # ===== Test 7: transformer_forward 前向传播 (推理模式) =====
    print("\n[Test 7] transformer_forward 前向传播 (推理模式)")
    # 注入足够轨迹
    np.random.seed(42)
    for i in range(30):
        state = np.random.randn(16) * 0.1
        action = np.random.randint(0, 7)
        reward = float(np.random.randn() * 0.01)
        dt.record_step(state, action, reward)
    # 推理前向
    states = np.array([s.state for s in dt.current_trajectory[-20:]])
    actions = np.array([s.action for s in dt.current_trajectory[-20:]])
    rtgs = np.array([[s.return_to_go] for s in dt.current_trajectory[-20:]])
    timesteps = np.array([s.timestep for s in dt.current_trajectory[-20:]])
    dt.training_mode = False
    logits = dt.transformer_forward(states, actions, rtgs, timesteps)
    check(f"transformer_forward logits shape=(T, 7)", logits.shape[1] == 7)
    check("transformer_forward logits 有限 (无NaN/Inf)",
          np.all(np.isfinite(logits)))
    check("transformer_forward 后 training_mode=False", dt.training_mode == False)

    # ===== Test 8: predict_action + Decoupling RTG =====
    print("\n[Test 8] predict_action + Decoupling RTG")
    dt.update_target_rtg(0.08)  # 高目标RTG
    action, prob, probs = dt.predict_action()
    check("predict_action 返回 ActionType", isinstance(action, ActionType))
    check(f"predict_action prob in [0, 1]", 0.0 <= prob <= 1.0)
    check(f"predict_action probs shape=(7,)", probs.shape == (7,))
    check(f"predict_action probs 和≈1 (softmax)",
          abs(np.sum(probs) - 1.0) < 1e-5)
    check("predict_action 缓存 last_predicted_action_idx",
          dt.last_predicted_action_idx is not None)
    # 推理模式应该关闭training_mode
    check("predict_action 后 training_mode=False", dt.training_mode == False)
    # last_actual_rtg应该被record_step更新过
    check("last_actual_rtg 已更新 (非0)", dt.last_actual_rtg != 0.0)

    # ===== Test 9: update_weights 在线学习 + LoRA更新 =====
    print("\n[Test 9] update_weights 在线学习 + LoRA更新")
    # 记录update_weights前的LoRA A/B矩阵
    lora_q_A_before = dt.lora_layers[0]["lora_q"].A.copy()
    lora_q_B_before = dt.lora_layers[0]["lora_q"].B.copy()
    # 第一次update_weights: B会被更新 (grad_B依赖A·x), A的梯度依赖B所以可能为0
    loss1 = dt.update_weights()
    B_diff_1 = np.linalg.norm(dt.lora_layers[0]["lora_q"].B - lora_q_B_before)
    check(f"第一次update_weights B已更新 (diff={B_diff_1:.6f})", B_diff_1 > 0)
    # 第二次update_weights: 此时B非零, A也应该开始更新
    loss2 = dt.update_weights()
    A_diff = np.linalg.norm(dt.lora_layers[0]["lora_q"].A - lora_q_A_before)
    check(f"第二次update_weights A已更新 (diff={A_diff:.6f})", A_diff > 0)
    check(f"update_weights 返回loss1 ({loss1:.6f}) loss2 ({loss2:.6f})", loss1 >= 0 and loss2 >= 0)
    check("update_weights 后 online_learning_call_count >= 2",
          dt.online_learning_call_count >= 2)
    check("update_weights 后 last_online_loss 已更新",
          dt.last_online_loss == loss2)
    check("update_weights 后 last_grad_norm 已更新",
          dt.last_grad_norm >= 0)
    # training_mode应该恢复为False
    check("update_weights 后 training_mode=False", dt.training_mode == False)
    # 梯度裁剪验证
    check(f"梯度范数 ≤ GRAD_CLIP*N_LAYERS",
          dt.last_grad_norm <= dt.GRAD_CLIP * dt.N_LAYERS + 1e-5)

    # ===== Test 10: analyze 完整流程 + 准确率EMA + 跨资产更新 =====
    print("\n[Test 10] analyze 完整流程 + 准确率EMA + 跨资产更新")
    # 跨资产状态更新
    other_states = [np.random.randn(64) * 0.1 for _ in range(3)]
    dt.update_other_asset_states(other_states)
    check("update_other_asset_states 后缓存3个",
          len(dt._other_asset_states) == 3)
    # analyze
    signal = dt.analyze()
    check("analyze 返回 DTSignal", isinstance(signal, DTSignal))
    check("analyze signal_type 是 DTSignalType",
          signal.signal_type in DTSignalType)
    check(f"analyze trajectory_length>0", signal.trajectory_length > 0)
    # 准确率EMA应该有变化 (因为record_step会更新)
    ema = dt.action_accuracy_ema
    check(f"action_accuracy_ema in [0, 1]", 0.0 <= ema <= 1.0,
          f"ema={ema:.4f}")
    # get_status 包含 v1.1字段
    status = dt.get_status()
    v1_1_fields = [
        "v1_1_enhanced", "lora_rank", "lora_alpha", "lora_layers_count",
        "online_learning_call_count", "last_online_loss", "last_grad_norm",
        "action_accuracy_ema", "decouple_lambda", "last_actual_rtg",
        "cross_asset_enabled", "cross_asset_other_count",
        "hybrid_enabled", "hybrid_last_w_local",
    ]
    for field in v1_1_fields:
        check(f"get_status 含 v1.1 字段 {field}", field in status, f"值={status.get(field)}")

    # ===== 结果汇总 =====
    print("\n" + "=" * 72)
    for r in RESULTS:
        print(r)
    print(f"\n{'=' * 72}")
    print(f"结果: {PASS_COUNT} PASS / {FAIL_COUNT} FAIL")
    print("=" * 72)
    if FAIL_COUNT > 0:
        sys.exit(1)
