# -*- coding: utf-8 -*-
"""时序预测增强模块 (Time-Series Forecasting Enhancement)

诚实描述：
    本模块是一个线性基线（多尺度统计 + FFT 频域 + 指数衰减加权"注意力"）
    加上可选的 PyTorch 轻量增强（TFT 风格小型网络），用于给交易决策
    系统提供一个方向预测与置信度参考信号。

    本模块不是任何具体论文（TFT/MSTFNet/FEDformer 等）的实现，所有
    "论文引用"、"56.3% 准确率" 等无验证描述均已删除。任何具体准确率
    数字在没有回测验证前都不应被引用。

主要组件：
    1. 线性基线 (LinearBaseline) — 公开类
       - 多尺度滑动统计 (short/mid/long)
       - FFT 频域主周期检测
       - 指数衰减 + 波动率加权的"注意力"（非真 softmax 自注意力）
       - 在线 EMA 准确率跟踪
    2. PyTorch 轻量增强 (TorchEnhancer，默认关闭，enable_torch=True 启用)
       - _TinyTFT: Linear → 1-layer LSTM → 单头自注意力 → GRN → Linear
       - 参数量 < 50K，CPU 推理 < 5ms
       - 离线训练 + 在线推理，与线性基线双路加权融合

设计原则：
    - 默认纯 numpy 实现，零外部依赖
    - PyTorch 懒加载（try import），不可用时降级为纯 LinearBaseline
    - 在线增量更新
    - 输出方向预测 + 置信度，作为信号增强而非直接决策
    - 反过拟合：使用简单模型 + 正则化，避免过深网络

兼容性说明：
    原 DirectionPredictor / TemporalAttention / MultiScaleFeatureExtractor /
    FrequencyDomainEnhancer 类已被重命名为更诚实的名字：
        DirectionPredictor           → LinearBaseline           (公开类)
        TemporalAttention            → _ExpDecayAttention       (内部类)
        MultiScaleFeatureExtractor   → _MultiScaleFeatures      (内部类)
        FrequencyDomainEnhancer      → _FFTEnhancer             (内部类)
    DeepLearningForecastSystem.update() 返回原全部 10 个键，零破坏性。
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("hermes.dl_forecast")


# ============================================================================
# 1. 多尺度时序特征提取 (内部类)
# ============================================================================


class _MultiScaleFeatures:
    """多尺度时序特征提取器（线性统计）

    纯 numpy 实现：用不同窗口大小的滑动平均 + 差分 / 斜率，提取多尺度
    趋势特征。无任何论文引用，无准确率承诺。

    量化阈值：
        - 短期窗口: 5 根 K 线
        - 中期窗口: 20 根 K 线
        - 长期窗口: 60 根 K 线
    """

    def __init__(
        self,
        short_window: int = 5,
        mid_window: int = 20,
        long_window: int = 60,
    ):
        self.short_window = short_window
        self.mid_window = mid_window
        self.long_window = long_window
        self.price_history: deque = deque(maxlen=long_window + 10)

    def update(self, price: float) -> Dict[str, float]:
        """更新价格历史并提取多尺度特征

        Returns:
            包含短期/中期/长期趋势特征的字典
        """
        self.price_history.append(price)
        prices = np.array(self.price_history)

        features: Dict[str, float] = {}

        # 短期特征
        if len(prices) >= self.short_window:
            short_ma = np.mean(prices[-self.short_window:])
            short_std = np.std(prices[-self.short_window:])
            features["short_ma"] = float(short_ma)
            features["short_momentum"] = float(
                (price - short_ma) / (short_std + 1e-8)
            )
            features["short_slope"] = float(
                np.polyfit(range(self.short_window), prices[-self.short_window:], 1)[0]
                if self.short_window >= 2
                else 0.0
            )
        else:
            features.update(
                {"short_ma": price, "short_momentum": 0.0, "short_slope": 0.0}
            )

        # 中期特征
        if len(prices) >= self.mid_window:
            mid_ma = np.mean(prices[-self.mid_window:])
            mid_std = np.std(prices[-self.mid_window:])
            features["mid_ma"] = float(mid_ma)
            features["mid_momentum"] = float((price - mid_ma) / (mid_std + 1e-8))
            features["mid_slope"] = float(
                np.polyfit(range(self.mid_window), prices[-self.mid_window:], 1)[0]
            )
        else:
            features.update({"mid_ma": price, "mid_momentum": 0.0, "mid_slope": 0.0})

        # 长期特征
        if len(prices) >= self.long_window:
            long_ma = np.mean(prices[-self.long_window:])
            long_std = np.std(prices[-self.long_window:])
            features["long_ma"] = float(long_ma)
            features["long_momentum"] = float((price - long_ma) / (long_std + 1e-8))
            features["long_slope"] = float(
                np.polyfit(range(self.long_window), prices[-self.long_window:], 1)[0]
            )
        else:
            features.update(
                {"long_ma": price, "long_momentum": 0.0, "long_slope": 0.0}
            )

        # 多尺度共振：三尺度方向是否一致
        short_dir = 1 if features["short_slope"] > 0 else -1
        mid_dir = 1 if features["mid_slope"] > 0 else -1
        long_dir = 1 if features["long_slope"] > 0 else -1
        features["scale_resonance"] = float(
            (short_dir + mid_dir + long_dir) / 3.0
        )  # [-1, 1]

        return features


# ============================================================================
# 2. 指数衰减"注意力" (内部类，非真自注意力)
# ============================================================================


class _ExpDecayAttention:
    """指数衰减 + 波动率加权的"注意力"（非真 softmax 自注意力）

    注意：这不是 TFT / Transformer 中的自注意力机制，仅是一个基于
    指数时间衰减 + 波动率加权的加权平均。原类名 TemporalAttention
    误导性较强，已重命名为 _ExpDecayAttention 以反映真实实现。

    量化：
        - 衰减系数: 0.95（每根 K 线衰减 5%）
        - 波动率权重: 高波动时段权重提升 1.0-1.5 倍
        - softmax 归一化
    """

    def __init__(
        self,
        window_size: int = 20,
        decay_factor: float = 0.95,
        vol_weight_strength: float = 0.5,
    ):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.vol_weight_strength = vol_weight_strength
        self.returns_history: deque = deque(maxlen=window_size + 5)

    def update_and_attend(
        self, returns: float, volatility: float
    ) -> Dict[str, Any]:
        """更新历史并计算加权特征

        Args:
            returns: 当期收益率
            volatility: 当期波动率（年化）

        Returns:
            加权后的特征字典
        """
        self.returns_history.append(returns)

        if len(self.returns_history) < 2:
            return {
                "attended_return": returns,
                "attention_entropy": 0.0,
                "focus_weight": 1.0,
            }

        returns_arr = np.array(list(self.returns_history)[-self.window_size:])

        n = len(returns_arr)
        # 1. 时间衰减权重
        time_weights = np.array(
            [self.decay_factor ** (n - 1 - i) for i in range(n)]
        )

        # 2. 波动率权重
        abs_returns = np.abs(returns_arr)
        vol_weights = 1.0 + self.vol_weight_strength * (
            abs_returns / (np.mean(abs_returns) + 1e-8)
        )
        vol_weights = np.clip(vol_weights, 1.0, 1.5)

        # 3. 组合权重并 softmax 归一化
        combined_weights = time_weights * vol_weights
        combined_weights = combined_weights / (np.sum(combined_weights) + 1e-8)

        attended_return = float(np.sum(combined_weights * returns_arr))

        attention_entropy = float(
            -np.sum(combined_weights * np.log(combined_weights + 1e-8))
        )
        max_entropy = math.log(n) if n > 1 else 1.0
        attention_entropy = attention_entropy / max_entropy if max_entropy > 0 else 0.0

        focus_weight = 1.0 + (1.0 - attention_entropy) * 0.3

        return {
            "attended_return": attended_return,
            "attention_entropy": attention_entropy,
            "focus_weight": float(focus_weight),
            "weight_distribution": combined_weights.tolist(),
        }


# ============================================================================
# 3. FFT 频域增强 (内部类)
# ============================================================================


class _FFTEnhancer:
    """FFT 频域特征增强器（numpy）

    原理：FFT 提取频域主周期，输出周期相位作为信号。
    本实现没有论文背书，仅作为线性基线的一个补充特征源。

    量化：
        - 最小周期: 5 根 K 线
        - 最大周期: 60 根 K 线
        - 主周期检测: 功率谱密度 top-1
        - 频域置信度: 主周期能量占比
    """

    def __init__(
        self,
        window_size: int = 60,
        min_period: int = 5,
        max_period: int = 60,
    ):
        self.window_size = window_size
        self.min_period = min_period
        self.max_period = max_period
        self.price_history: deque = deque(maxlen=window_size + 5)

    def update_and_analyze(self, price: float) -> Dict[str, Any]:
        """更新价格并分析频域特征

        Returns:
            频域特征字典，包含主周期、频域置信度等
        """
        self.price_history.append(price)

        if len(self.price_history) < self.min_period * 2:
            return {
                "dominant_period": 0.0,
                "spectral_energy": 0.0,
                "freq_confidence": 0.0,
                "freq_signal": 0.0,
            }

        prices = np.array(list(self.price_history)[-self.window_size:])

        # 去趋势（减去均值）
        prices_detrended = prices - np.mean(prices)

        # FFT
        fft_result = np.fft.rfft(prices_detrended)
        power_spectrum = np.abs(fft_result) ** 2

        n = len(prices_detrended)
        freqs = np.fft.rfftfreq(n, d=1.0)

        # 过滤频率范围（对应周期 5-60）
        valid_mask = (
            (freqs >= 1.0 / self.max_period) & (freqs <= 1.0 / self.min_period)
        )
        valid_power = power_spectrum[valid_mask]
        valid_freqs = freqs[valid_mask]

        if len(valid_power) == 0 or np.sum(valid_power) < 1e-8:
            return {
                "dominant_period": 0.0,
                "spectral_energy": 0.0,
                "freq_confidence": 0.0,
                "freq_signal": 0.0,
            }

        # 主周期（功率最大的频率）
        max_idx = np.argmax(valid_power)
        dominant_freq = valid_freqs[max_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0

        # 频域能量占比
        total_energy = np.sum(power_spectrum) + 1e-8
        dominant_energy_ratio = float(valid_power[max_idx] / total_energy)

        freq_confidence = min(1.0, dominant_energy_ratio * 5.0)

        # 频域信号：基于主周期相位
        dominant_phase = np.angle(fft_result[valid_mask][max_idx])
        freq_signal = float(math.cos(dominant_phase))

        return {
            "dominant_period": float(dominant_period),
            "spectral_energy": float(np.sum(valid_power)),
            "freq_confidence": float(freq_confidence),
            "freq_signal": freq_signal,  # [-1, 1]，正=上升期，负=下降期
        }


# ============================================================================
# 4. 线性基线方向预测器 (公开类)
# ============================================================================


@dataclass
class ForecastResult:
    """预测结果"""

    direction: int  # 1=上涨, -1=下跌, 0=中性
    probability: float  # 方向概率 [0.5, 1.0]
    confidence: float  # 置信度 [0, 1]
    predicted_return: float  # 预测收益率
    horizon: int  # 预测视野（K线数）
    components: Dict[str, float] = field(default_factory=dict)


class LinearBaseline:
    """线性基线方向预测器（公开类）

    原 DirectionPredictor 的重命名版本。融合多尺度统计 + 指数衰减注意力 +
    FFT 频域特征，输出方向预测 + 连续置信度。

    本类不引用任何论文，不承诺任何准确率数字。
    所有 "56.3%"、"Google 2021"、"FEDformer 2022" 等描述已删除。

    融合策略（默认权重）：
        1. 多尺度共振信号 (权重 0.4)
        2. 指数衰减注意力加权收益 (权重 0.3)
        3. FFT 频域信号 (权重 0.3)

    Bug 修复：
        - enable_attention / enable_frequency 开关现在真正控制是否创建对应
          特征器（原版本中开关被读取但不影响行为）
        - confidence 从离散值 {0, 1/3, 2/3, 1} 改为连续值 [0, 1]，使 0.2
          阈值不再仅在"完全不一致"时才触发（原 bug）
    """

    def __init__(
        self,
        prediction_horizon: int = 5,
        min_confidence: float = 0.55,
        enable_attention: bool = True,
        enable_frequency: bool = True,
    ):
        self.prediction_horizon = prediction_horizon
        self.min_confidence = min_confidence
        self.enable_attention = enable_attention
        self.enable_frequency = enable_frequency

        # 三个特征提取器（修复 bug：开关真正控制创建）
        self.multi_scale = _MultiScaleFeatures()
        self.attention: Optional[_ExpDecayAttention] = (
            _ExpDecayAttention() if enable_attention else None
        )
        self.frequency: Optional[_FFTEnhancer] = (
            _FFTEnhancer() if enable_frequency else None
        )

        # 在线学习：维护预测准确率
        self.prediction_history: deque = deque(maxlen=100)
        self.accuracy_ema: float = 0.5  # 指数移动平均准确率

    def update(self, price: float, volatility: float = 0.0) -> ForecastResult:
        """更新模型并生成预测

        Args:
            price: 当前价格
            volatility: 当前波动率（年化，可选）

        Returns:
            ForecastResult 预测结果
        """
        # 计算收益率
        returns = 0.0
        if hasattr(self, "_last_price") and self._last_price > 0:
            returns = (price - self._last_price) / self._last_price
        self._last_price = price

        # 提取特征（受开关控制）
        ms_features = self.multi_scale.update(price)
        if self.attention is not None:
            attn_features = self.attention.update_and_attend(returns, volatility)
        else:
            attn_features = {
                "attended_return": returns,
                "focus_weight": 1.0,
                "attention_entropy": 0.0,
            }
        if self.frequency is not None:
            freq_features = self.frequency.update_and_analyze(price)
        else:
            freq_features = {
                "freq_signal": 0.0,
                "freq_confidence": 0.0,
            }

        # ---- 融合得分 ----
        # 1. 多尺度共振信号
        scale_signal = ms_features.get("scale_resonance", 0.0)
        short_mom = ms_features.get("short_momentum", 0.0)
        mid_mom = ms_features.get("mid_momentum", 0.0)
        scale_score = (
            scale_signal * 0.5
            + np.tanh(short_mom) * 0.25
            + np.tanh(mid_mom) * 0.25
        )

        # 2. 注意力加权收益
        attended_return = attn_features.get("attended_return", 0.0)
        focus_weight = attn_features.get("focus_weight", 1.0)
        attn_score = float(np.tanh(attended_return * 50)) * focus_weight

        # 3. 频域信号
        freq_signal = freq_features.get("freq_signal", 0.0)
        freq_conf = freq_features.get("freq_confidence", 0.0)
        freq_score = freq_signal * freq_conf

        # 融合（加权求和）—— 当某些特征器被关闭时，重新归一化权重
        if self.attention is None and self.frequency is None:
            fused_score = scale_score
        elif self.attention is None:
            fused_score = scale_score * 0.6 + freq_score * 0.4
        elif self.frequency is None:
            fused_score = scale_score * 0.6 + attn_score * 0.4
        else:
            fused_score = (
                scale_score * 0.4
                + attn_score * 0.3
                + freq_score * 0.3
            )

        # 方向概率：sigmoid
        probability = 1.0 / (1.0 + math.exp(-fused_score * 5.0))

        # 方向判断
        if probability > self.min_confidence:
            direction = 1
        elif probability < (1.0 - self.min_confidence):
            direction = -1
        else:
            direction = 0

        # ---- 置信度（连续值，修复离散 bug） ----
        # 三源归一化方向得分
        s1 = float(np.tanh(scale_score))
        s2 = float(np.tanh(attn_score))
        s3 = float(np.tanh(freq_score))
        # 一致性 = 三源符号平均的绝对值 ∈ [0, 1]
        #   完全一致=1，弱一致=1/3 或 2/3，完全不一致=0
        signs = np.array([np.sign(s1), np.sign(s2), np.sign(s3)])
        consistency = float(abs(signs.mean()))
        # 强度 = 平均绝对值（归一化到 [0, 1]）
        strength = float((abs(s1) + abs(s2) + abs(s3)) / 3.0)
        strength = min(1.0, strength * 2.0)
        # 综合 confidence：一致性主导，强度作调节 ∈ [0, 1]
        #   完全不一致 → 0
        #   弱一致 + 低强度 → ~0.13（< 0.2，触发阈值）
        #   完全一致 + 高强度 → 1.0
        confidence = float(consistency * (0.4 + 0.6 * strength))

        # 预测收益率
        predicted_return = fused_score * 0.02

        # 记录预测用于在线学习
        self.prediction_history.append(
            {"direction": direction, "probability": probability, "returns": returns}
        )

        # 更新准确率 EMA
        if len(self.prediction_history) >= 10:
            recent = list(self.prediction_history)[-10:]
            correct = sum(
                1
                for p in recent
                if (p["direction"] == 1 and p["returns"] > 0)
                or (p["direction"] == -1 and p["returns"] < 0)
            )
            recent_accuracy = correct / len(recent)
            self.accuracy_ema = (
                0.9 * self.accuracy_ema + 0.1 * recent_accuracy
            )

        return ForecastResult(
            direction=direction,
            probability=float(probability),
            confidence=float(confidence),
            predicted_return=float(predicted_return),
            horizon=self.prediction_horizon,
            components={
                "scale_score": float(scale_score),
                "attn_score": float(attn_score),
                "freq_score": float(freq_score),
                "fused_score": float(fused_score),
                "accuracy_ema": float(self.accuracy_ema),
                "consistency": float(consistency),
                "strength": float(strength),
                "enable_attention": 1.0 if self.attention is not None else 0.0,
                "enable_frequency": 1.0 if self.frequency is not None else 0.0,
            },
        )


# ============================================================================
# 5. 可选 PyTorch 轻量增强 (默认关闭)
# ============================================================================


class _TinyTFT:
    """轻量 TFT 风格网络（懒加载 PyTorch）

    结构: Linear embedding → 1-layer LSTM → 单头自注意力 → GRN → Linear
    参数量 < 50K，CPU 推理 < 5ms

    本类不是 Google TFT 的完整实现，仅是受其启发的小型网络，用于
    与 LinearBaseline 双路融合。torch 不可用时 model 为 None。
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, seq_len: int = 20):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self._torch = None
        self._nn = None
        self.model = None  # type: ignore[assignment]
        self._build()

    def _build(self) -> None:
        """构建网络（torch 不可用时降级为 None）"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self._torch = None
            self._nn = None
            self.model = None
            return

        self._torch = torch
        self._nn = nn

        hidden = self.hidden_dim
        input_dim = self.input_dim

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                # 1. Linear embedding
                self.input_proj = nn.Linear(input_dim, hidden)
                # 2. 1-layer LSTM
                self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
                # 3. 单头自注意力
                self.q_proj = nn.Linear(hidden, hidden)
                self.k_proj = nn.Linear(hidden, hidden)
                self.v_proj = nn.Linear(hidden, hidden)
                self.attn_scale = float(hidden) ** 0.5
                # 4. GRN (Gated Residual Network)
                self.grn_fc1 = nn.Linear(hidden, hidden * 2)
                self.grn_gate = nn.Linear(hidden * 2, hidden * 2)
                self.grn_fc2 = nn.Linear(hidden * 2, hidden)
                # LayerNorm
                self.ln1 = nn.LayerNorm(hidden)
                self.ln2 = nn.LayerNorm(hidden)
                # 5. 输出
                self.output_proj = nn.Linear(hidden, 1)

            def forward(self, x):
                # x: [B, T, input_dim]
                h = self.input_proj(x)  # [B, T, H]
                lstm_out, _ = self.lstm(h)  # [B, T, H]
                # 单头自注意力 over T
                q = self.q_proj(lstm_out)
                k = self.k_proj(lstm_out)
                v = self.v_proj(lstm_out)
                attn = torch.matmul(q, k.transpose(-2, -1)) / self.attn_scale
                attn = torch.softmax(attn, dim=-1)
                ctx = torch.matmul(attn, v)  # [B, T, H]
                ctx = self.ln1(ctx + lstm_out)
                # GRN: ELU → GLU → Linear → 残差
                g = self.grn_fc1(ctx)  # [B, T, 2H]
                g = torch.elu(g)
                gate = torch.sigmoid(self.grn_gate(g))
                g = g * gate
                g = self.grn_fc2(g)  # [B, T, H]
                grn_out = self.ln2(ctx + g)
                # 输出最后一帧
                last = grn_out[:, -1, :]
                return self.output_proj(last)  # [B, 1]

        self.model = _Impl()
        # 评估模式（推理时）
        self.model.eval()

    def param_count(self) -> int:
        """返回模型参数量"""
        if self._torch is None or self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))

    def predict(self, x_np: np.ndarray) -> Optional[float]:
        """前向推理

        Args:
            x_np: [T, input_dim] numpy 数组

        Returns:
            预测标量，torch 不可用时返回 None
        """
        if self._torch is None or self.model is None:
            return None
        torch = self._torch
        self.model.eval()
        # x_np: [T, input_dim] → [1, T, input_dim]
        x = torch.from_numpy(x_np.astype(np.float32)).view(1, -1, self.input_dim)
        with torch.no_grad():
            out = self.model(x)
        return float(out.item())


class TorchEnhancer:
    """PyTorch 轻量增强器（默认关闭，懒加载）

    离线训练 + 在线推理，与 LinearBaseline 双路融合。
    torch 不可用时降级为 noop（update 返回 None）。

    使用：
        enhancer = TorchEnhancer()
        enhancer.fit(price_array)            # 离线训练
        result = enhancer.update(price, vol)  # 在线推理
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_dim: int = 16,
        lr: float = 1e-3,
        train_interval: int = 50,
        train_epochs: int = 20,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.train_interval = train_interval
        self.train_epochs = train_epochs
        self.price_history: deque = deque(maxlen=seq_len + 200)
        self.model: Optional[_TinyTFT] = None
        self.optimizer = None
        self.criterion = None
        self._torch = None
        self._nn = None
        self._is_trained = False
        self._update_count = 0
        self._last_prediction = 0.0
        self._init_model()

    def _init_model(self) -> None:
        """初始化模型与优化器（torch 不可用时降级）"""
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn = nn
        except ImportError:
            self._torch = None
            self._nn = None
            self.model = None
            logger.info("TorchEnhancer: torch not available, fallback to noop")
            return

        self.model = _TinyTFT(
            input_dim=1, hidden_dim=self.hidden_dim, seq_len=self.seq_len
        )
        if self.model.model is None:
            # _TinyTFT 内部构建失败
            self.model = None
            return
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.model.model.eval()
        logger.info(
            "TorchEnhancer: initialized _TinyTFT, params=%d",
            self.model.param_count(),
        )

    def is_available(self) -> bool:
        """torch 是否可用"""
        return self._torch is not None and self.model is not None

    def fit(self, price_array: np.ndarray) -> bool:
        """离线训练

        用过去 seq_len 步预测下一步的 log price。
        训练数据通过对数价格差分构造。

        Returns:
            训练成功返回 True，否则 False
        """
        if not self.is_available():
            return False
        price_array = np.asarray(price_array, dtype=np.float64)
        if len(price_array) < self.seq_len + 5:
            return False

        torch = self._torch
        model = self.model.model  # type: ignore[union-attr]
        model.train()

        # 对数价格归一化
        log_prices = np.log(price_array + 1e-8)
        X, Y = [], []
        for i in range(len(log_prices) - self.seq_len):
            X.append(log_prices[i:i + self.seq_len].reshape(-1, 1))
            Y.append(log_prices[i + self.seq_len])
        X_arr = np.array(X, dtype=np.float32)
        Y_arr = np.array(Y, dtype=np.float32)
        if len(X_arr) == 0:
            return False

        X_t = torch.from_numpy(X_arr)
        Y_t = torch.from_numpy(Y_arr).view(-1, 1)

        try:
            for _ in range(self.train_epochs):
                self.optimizer.zero_grad()
                pred = model(X_t)
                loss = self.criterion(pred, Y_t)
                loss.backward()
                self.optimizer.step()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TorchEnhancer fit failed: %s", exc)
            model.eval()
            return False

        model.eval()
        self._is_trained = True
        return True

    def update(self, price: float, volatility: float = 0.0) -> Optional[Dict[str, Any]]:
        """在线推理

        Returns:
            包含 direction / predicted_return / confidence 的字典，
            torch 不可用或样本不足时返回 None
        """
        if not self.is_available():
            return None

        self.price_history.append(price)
        self._update_count += 1

        # 定期重新训练
        if (
            self._update_count % self.train_interval == 0
            and len(self.price_history) >= self.seq_len + 10
        ):
            try:
                self.fit(np.array(list(self.price_history), dtype=np.float64))
            except Exception as exc:  # noqa: BLE001
                logger.debug("TorchEnhancer periodic refit failed: %s", exc)

        if not self._is_trained or len(self.price_history) < self.seq_len:
            return None

        # 推理：用最近 seq_len 个价格
        recent = np.array(list(self.price_history)[-self.seq_len:], dtype=np.float64)
        log_recent = np.log(recent + 1e-8).reshape(-1, 1)
        pred_log = self.model.predict(log_recent)  # type: ignore[union-attr]
        if pred_log is None:
            return None

        current_log = float(np.log(price + 1e-8))
        predicted_return = float(pred_log - current_log)
        self._last_prediction = predicted_return

        # 方向
        if predicted_return > 1e-6:
            direction = 1
        elif predicted_return < -1e-6:
            direction = -1
        else:
            direction = 0

        # 简化 confidence：基于 |predicted_return| 与 volatility 的比
        vol = max(volatility, 1e-4)
        conf = min(1.0, abs(predicted_return) / (vol + 1e-8) * 0.5)

        return {
            "direction": direction,
            "predicted_return": predicted_return,
            "confidence": float(conf),
        }


# ============================================================================
# 6. 时序预测综合系统
# ============================================================================


class DeepLearningForecastSystem:
    """时序预测综合系统

    整合线性基线 + 可选 PyTorch 轻量增强的方向预测系统。

    使用方式：
        system = DeepLearningForecastSystem()
        result = system.update(price, volatility)
        if result["should_enhance"]:
            signal_score *= result["enhancement_factor"]

    接口兼容性：
        update() 返回 10 个键（详见返回 docstring），默认 enable_torch=False
        保持与原版本完全兼容；enable_torch=True 启用 PyTorch 增强双路融合。
    """

    def __init__(
        self,
        prediction_horizon: int = 5,
        min_confidence: float = 0.55,
        enable_attention: bool = True,
        enable_frequency: bool = True,
        enable_torch: bool = False,
    ):
        self.predictor = LinearBaseline(
            prediction_horizon=prediction_horizon,
            min_confidence=min_confidence,
            enable_attention=enable_attention,
            enable_frequency=enable_frequency,
        )
        self.enable_attention = enable_attention
        self.enable_frequency = enable_frequency
        self.enable_torch = enable_torch

        # PyTorch 增强器（懒加载，默认 None）
        self.enhancer: Optional[TorchEnhancer] = None
        if enable_torch:
            self.enhancer = TorchEnhancer()
            if not self.enhancer.is_available():
                # torch 不可用，降级
                logger.warning(
                    "enable_torch=True but torch unavailable, "
                    "fallback to pure LinearBaseline"
                )
                self.enhancer = None

        # 状态
        self.last_result: Optional[ForecastResult] = None
        self.update_count: int = 0

    def update(
        self, price: float, volatility: float = 0.0
    ) -> Dict[str, Any]:
        """更新预测系统

        Returns:
            完整的预测状态字典，包含 10 个键：
                - direction: 预测方向 (1/-1/0)
                - direction_label: "bullish" / "bearish" / "neutral"
                - probability: 方向概率 [0.5, 1.0]
                - confidence: 置信度 [0, 1]
                - predicted_return: 预测收益率
                - horizon: 预测视野（K线数）
                - enhancement_factor: 信号增强因子 (0.7-1.3)
                - should_enhance: 是否应该增强信号
                - components: 各组件得分
                - accuracy_ema: 准确率 EMA
                - update_count: 更新次数（可选附加键）
        """
        self.update_count += 1

        result = self.predictor.update(price, volatility)
        self.last_result = result

        # PyTorch 增强器双路融合
        enhancer_result = None
        enh_blend = 0.0
        if self.enhancer is not None:
            enhancer_result = self.enhancer.update(price, volatility)

        if enhancer_result is not None:
            # 加权融合：baseline 0.7 + enhancer 0.3
            enh_blend = 0.3
            base_return = result.predicted_return
            enh_return = enhancer_result["predicted_return"]
            blended_return = float(
                base_return * (1.0 - enh_blend) + enh_return * enh_blend
            )
            # 方向同步对 confidence 的影响
            if (
                enhancer_result["direction"] == result.direction
                and result.direction != 0
            ):
                # 一致 → 提升 confidence
                confidence = float(min(1.0, result.confidence + 0.1 * enh_blend))
            elif (
                enhancer_result["direction"] != 0
                and result.direction != 0
                and enhancer_result["direction"] != result.direction
            ):
                # 不一致 → 降低 confidence
                confidence = float(max(0.0, result.confidence - 0.15))
            else:
                confidence = result.confidence
            # 覆盖 result 字段
            result.predicted_return = blended_return
            result.confidence = confidence

        # ---- 计算信号增强因子 ----
        enhancement_factor = 1.0
        should_enhance = False

        if result.direction != 0 and result.confidence >= 0.6:
            # 高置信度预测：增强信号
            enhancement = (result.confidence - 0.6) / 0.4 * 0.3
            enhancement_factor = 1.0 + enhancement
            should_enhance = True
        elif result.confidence < 0.3:
            # 低置信度预测：减弱信号
            enhancement_factor = 0.7 + result.confidence
            should_enhance = True

        # 准确率 EMA 调节：模型近期表现差时降低增强幅度
        accuracy = result.components.get("accuracy_ema", 0.5)
        if accuracy < 0.5:
            enhancement_factor = 1.0 + (enhancement_factor - 1.0) * 0.5

        # 限制增强因子在 [0.7, 1.3]
        enhancement_factor = max(0.7, min(1.3, float(enhancement_factor)))

        # 组装 components
        components = dict(result.components)
        if enhancer_result is not None:
            components["enhancer_direction"] = float(enhancer_result["direction"])
            components["enhancer_confidence"] = float(enhancer_result["confidence"])
            components["enhancer_blend"] = float(enh_blend)
        components["enhancement_factor"] = enhancement_factor

        return {
            "direction": result.direction,
            "direction_label": (
                "bullish" if result.direction == 1
                else ("bearish" if result.direction == -1 else "neutral")
            ),
            "probability": result.probability,
            "confidence": result.confidence,
            "predicted_return": result.predicted_return,
            "horizon": result.horizon,
            "enhancement_factor": enhancement_factor,
            "should_enhance": should_enhance,
            "components": components,
            "accuracy_ema": accuracy,
            "update_count": self.update_count,
        }

    def get_signal_weight(self) -> float:
        """获取当前信号权重（用于决策引擎）

        Returns:
            信号权重 0.7-1.3
        """
        if self.last_result is None:
            return 1.0
        return self.last_result.components.get("enhancement_factor", 1.0)

    def get_state(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        if self.last_result is None:
            return {
                "status": "not_initialized",
                "update_count": 0,
                "enable_torch": self.enable_torch,
                "torch_active": self.enhancer is not None,
            }

        return {
            "status": "active",
            "update_count": self.update_count,
            "last_direction": self.last_result.direction,
            "last_confidence": self.last_result.confidence,
            "accuracy_ema": self.last_result.components.get("accuracy_ema", 0.5),
            "components": self.last_result.components,
            "enable_torch": self.enable_torch,
            "torch_active": self.enhancer is not None,
        }
