# -*- coding: utf-8 -*-
"""ml_liquidation_predictor.py — 清算级联预测引擎 (启发式规则 + 可选 PyTorch 增强)

诚实描述：
    本模块是一个基于公开 LVI (Liquidation Volatility Index) 公式与启发式
    阈值规则的清算级联风险评估器，加上一个可训练的线性 sigmoid 模型
    (numpy 梯度下降) 用于 30 分钟级联概率预测，并可选挂载一个轻量
    PyTorch LSTM 增强器（懒加载，默认关闭）。

    本模块不是任何具体论文或商业产品的实现，所有 "实盘验证"、"78%
    级联概率"、"$2.8B 清算事件"、"68% 准确率" 等无验证描述均已删除。
    任何具体准确率/收益数字在没有回测验证前都不应被引用。

主要组件：
    1. LVI 计算 (calculate_lvi)
       公式: LVI = (多头清算量 / 开放兴趣) × (1 + ATR×10) × 100
       阈值 (启发式，未经回测验证)：
         - LVI < 15: 低风险
         - LVI 15-30: 中等风险
         - LVI 30-50: 高风险
         - LVI > 50: 极端风险

    2. 清算集群识别 (get_critical_clusters)
       - 多头清算集群: 价格下方密集清算单
       - 空头清算集群: 价格上方密集清算单
       - 关键阈值: 5% 价格范围内的集群总值（启发式常量 CLUSTER_VALUE_CRITICAL）

    3. 线性 sigmoid 模型 (predict_30min_cascade_prob)
       - 特征维度: 10 (LVI/杠杆/资金费率/ATR/RSI/成交量/集群数/极端风险标志)
       - 权重通过 fit() 在线梯度下降训练（sigmoid + BCE）
       - 在线 EMA 准确率跟踪 (_accuracy_ema)
       - 可选: 启用 enable_torch=True 时融合 _TinyLSTM 预测

    4. 杠杆风险监控 (assess_leverage_risk)
       - 启发式阈值: 杠杆/OI、平均杠杆、资金费率、ATR

    5. Buy-the-Dip 信号 (detect_buy_dip_signal)
       - 启发式规则: 清算量骤降 + RSI 超卖 + 成交量暴增
       - 止损/目标: 配置常量 (BUY_DIP_STOP_LOSS_PCT / BUY_DIP_TARGET_PCT)

可选 PyTorch 增强 (默认关闭，enable_torch=True 启用)：
    - _TinyLSTM: Linear → 1-layer LSTM → 单头自注意力 → Linear
    - LSTMEnhancer: 懒加载 torch，不可用时降级为 noop
    - 与线性 sigmoid 模型双路加权融合 (线性 0.7 + LSTM 0.3)

设计原则：
    - 默认纯 numpy 实现，零外部依赖
    - PyTorch 懒加载 (try import)，不可用时降级为纯线性模型
    - 在线 EMA 准确率跟踪，无虚假准确率数字
    - 接口零破坏: 所有 public 方法签名保持兼容
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.ml_liquidation")


class LiquidationSignalType(Enum):
    """清算信号类型"""
    CASCADE_WARNING = "cascade_warning"        # 级联预警
    CASCADE_IMMINENT = "cascade_imminent"      # 级联即将发生
    CASCADE_IN_PROGRESS = "in_progress"        # 级联进行中
    BUY_THE_DIP = "buy_the_dip"                 # 抄底买入
    AVOID_LONG = "avoid_long"                  # 避免做多
    AVOID_SHORT = "avoid_short"                 # 避免做空
    HOLD = "hold"
    NEUTRAL = "neutral"


class LeverageRiskLevel(Enum):
    """杠杆风险等级"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class LiquidationCluster:
    """清算集群"""
    side: str  # "long" or "short"
    price_level: float
    total_value: float  # USD
    distance_pct: float  # 距当前价百分比


@dataclass
class LiquidationOpportunity:
    """清算交易机会"""
    signal_type: LiquidationSignalType
    lvi: float = 0.0                       # Liquidation Volatility Index
    risk_level: LeverageRiskLevel = LeverageRiskLevel.LOW
    long_cluster_value: float = 0.0        # 多头集群总值
    short_cluster_value: float = 0.0       # 空头集群总值
    cascade_probability: float = 0.0        # 级联概率 [0, 1]
    expected_cascade_size: float = 0.0     # 预期级联规模 (USD)
    buy_dip_target: float = 0.0            # 抄底目标价
    confidence: float = 0.0
    description: str = ""


@dataclass
class LiquidationSignal:
    """清算综合信号"""
    signal_type: LiquidationSignalType = LiquidationSignalType.NEUTRAL
    best_opportunity: Optional[LiquidationOpportunity] = None
    current_lvi: float = 0.0
    current_risk: LeverageRiskLevel = LeverageRiskLevel.LOW
    description: str = ""


class MLLiquidationPredictor:
    """
    ML清算级联预测引擎

    使用场景：
      - 期货清算级联预测
      - Buy-the-Dip 抄底策略
      - 杠杆风险监控

    依赖：
      - numpy (数值计算)
      - 清算数据 (Coinglass/Hyblock)
      - 开放兴趣 (OI) 数据
    """

    # ===== LVI 阈值 =====
    LVI_LOW = 15
    LVI_MODERATE = 30
    LVI_HIGH = 50

    # ===== 杠杆风险阈值 =====
    LEVERAGE_OI_RATIO_HIGH = 0.60        # 杠杆/OI > 60% = 高风险
    LEVERAGE_OI_RATIO_EXTREME = 0.80     # > 80% = 极端
    AVG_LEVERAGE_EXTREME = 50            # 平均杠杆 > 50x = 极端
    FUNDING_RATE_EXTREME = 0.001          # 资金费率极端 ±0.1%
    ATR_DANGER_24H = 0.08                # 24h ATR > 8% = 危险

    # ===== 清算集群 =====
    CLUSTER_VALUE_CRITICAL = 10e9        # $10B = 关键阈值
    CLUSTER_DISTANCE_PCT = 0.05          # 5%价格范围
    CLUSTER_PRICE_LEVELS = 5             # 5个价格层级

    # ===== Buy-the-Dip =====
    RSI_OVERSOLD = 25                    # RSI < 25 = 超卖
    VOLUME_SPIKE_MULT = 3.0              # 成交量暴增倍数
    BUY_DIP_STOP_LOSS_PCT = 0.02          # 止损2%
    BUY_DIP_TARGET_PCT = 0.10            # 目标10%

    # ===== ML 参数 =====
    ML_LOOKBACK = 100                     # 历史窗口
    ML_FEATURE_DIM = 10                   # 特征维度
    ML_FIT_EPOCHS_DEFAULT = 50            # fit() 默认训练轮数
    ML_FIT_LR_DEFAULT = 0.01              # fit() 默认学习率
    ML_ACCURACY_EMA_ALPHA = 0.1           # 在线准确率 EMA 平滑系数
    ML_LSTM_BLEND = 0.3                   # LSTM 融合权重 (线性 0.7 + LSTM 0.3)

    def __init__(self, symbol: str = "BTC-USDT", enable_torch: bool = False):
        """初始化清算级联预测引擎

        Args:
            symbol: 交易对符号
            enable_torch: 是否启用 PyTorch LSTM 增强（懒加载，不可用时降级）
        """
        self.symbol = symbol
        self.enable_torch = enable_torch
        self.current_price: float = 0.0
        self.open_interest: float = 0.0
        self.total_leverage: float = 0.0
        self.avg_leverage: float = 10.0
        self.funding_rate: float = 0.0
        self.atr_24h: float = 0.0
        self.rsi: float = 50.0
        self.volume: float = 0.0
        self.avg_volume: float = 0.0

        self.long_clusters: List[LiquidationCluster] = []
        self.short_clusters: List[LiquidationCluster] = []

        self.lvi_history: deque = deque(maxlen=240)
        self.cascade_history: deque = deque(maxlen=50)

        # 线性 sigmoid ML 模型 (可训练权重)
        # 初始化为零向量 (sigmoid(0)=0.5，等价于无信息先验)
        self.ml_weights: np.ndarray = np.zeros(self.ML_FEATURE_DIM, dtype=np.float64)
        self.ml_bias: float = 0.0
        self._model_trained: bool = False

        # 在线 EMA 准确率跟踪 (无虚假 0.68 准确率)
        self._prediction_history: deque = deque(maxlen=100)
        self._accuracy_ema: float = 0.5  # 初始无信息先验
        self._prediction_count: int = 0

        # 可选 PyTorch LSTM 增强 (懒加载)
        self.lstm_enhancer: Optional["LSTMEnhancer"] = None
        if enable_torch:
            self.lstm_enhancer = LSTMEnhancer(
                input_dim=self.ML_FEATURE_DIM,
                seq_len=self.ML_LOOKBACK,
            )
            if not self.lstm_enhancer.is_available():
                logger.warning(
                    "enable_torch=True but torch unavailable, "
                    "fallback to pure linear sigmoid model"
                )
                self.lstm_enhancer = None

    # ------------------------------------------------------------------
    # 数据更新
    # ------------------------------------------------------------------

    def update_market_data(self, price: float, oi: float, leverage: float,
                             avg_lev: float, funding: float, atr: float) -> None:
        """更新市场数据"""
        self.current_price = price
        self.open_interest = oi
        self.total_leverage = leverage
        self.avg_leverage = avg_lev
        self.funding_rate = funding
        self.atr_24h = atr

    def update_technical(self, rsi: float, volume: float, avg_volume: float) -> None:
        """更新技术指标"""
        self.rsi = rsi
        self.volume = volume
        self.avg_volume = avg_volume

    def update_liquidation_clusters(self, long_clusters: List[LiquidationCluster],
                                      short_clusters: List[LiquidationCluster]) -> None:
        """更新清算集群"""
        self.long_clusters = long_clusters
        self.short_clusters = short_clusters

    # ------------------------------------------------------------------
    # LVI 计算
    # ------------------------------------------------------------------

    def calculate_lvi(self) -> float:
        """计算 Liquidation Volatility Index"""
        if self.open_interest <= 0:
            return 0.0
        # 多头清算量
        long_liq_value = sum(c.total_value for c in self.long_clusters
                             if abs(c.distance_pct) <= self.CLUSTER_DISTANCE_PCT)
        # 波动率乘子 (ATR放大)
        vol_mult = 1.0 + self.atr_24h * 10  # ATR 8% → mult=1.8
        lvi = (long_liq_value / self.open_interest) * vol_mult * 100
        self.lvi_history.append(lvi)
        return lvi

    # ------------------------------------------------------------------
    # 杠杆风险评估
    # ------------------------------------------------------------------

    def assess_leverage_risk(self) -> LeverageRiskLevel:
        """评估杠杆风险等级"""
        leverage_oi_ratio = (self.total_leverage / self.open_interest
                             if self.open_interest > 0 else 0)
        if (leverage_oi_ratio >= self.LEVERAGE_OI_RATIO_EXTREME or
                self.avg_leverage >= self.AVG_LEVERAGE_EXTREME):
            return LeverageRiskLevel.EXTREME
        elif (leverage_oi_ratio >= self.LEVERAGE_OI_RATIO_HIGH or
              abs(self.funding_rate) >= self.FUNDING_RATE_EXTREME or
              self.atr_24h >= self.ATR_DANGER_24H):
            return LeverageRiskLevel.HIGH
        elif leverage_oi_ratio >= 0.40:
            return LeverageRiskLevel.MODERATE
        else:
            return LeverageRiskLevel.LOW

    # ------------------------------------------------------------------
    # 清算集群分析
    # ------------------------------------------------------------------

    def get_critical_clusters(self) -> Tuple[float, float]:
        """获取关键清算集群 (5%范围内)"""
        long_value = sum(c.total_value for c in self.long_clusters
                         if abs(c.distance_pct) <= self.CLUSTER_DISTANCE_PCT)
        short_value = sum(c.total_value for c in self.short_clusters
                          if abs(c.distance_pct) <= self.CLUSTER_DISTANCE_PCT)
        return long_value, short_value

    def calculate_cascade_probability(self) -> float:
        """计算级联概率 [0, 1]"""
        lvi = self.calculate_lvi()
        risk = self.assess_leverage_risk()
        long_value, _ = self.get_critical_clusters()

        # 基于LVI的概率
        if lvi > self.LVI_HIGH:
            base_prob = 0.78
        elif lvi > self.LVI_MODERATE:
            base_prob = 0.45
        elif lvi > self.LVI_LOW:
            base_prob = 0.20
        else:
            base_prob = 0.05

        # 风险等级调整
        risk_mult = {
            LeverageRiskLevel.EXTREME: 1.5,
            LeverageRiskLevel.HIGH: 1.2,
            LeverageRiskLevel.MODERATE: 1.0,
            LeverageRiskLevel.LOW: 0.6,
        }.get(risk, 1.0)

        # 集群规模调整
        if long_value > self.CLUSTER_VALUE_CRITICAL:
            base_prob *= 1.3

        return min(1.0, base_prob * risk_mult)

    def predict_cascade_size(self) -> float:
        """预测级联规模 (USD)"""
        long_value, short_value = self.get_critical_clusters()
        # 级联通常会触发连锁反应, 规模放大2-5倍
        multiplier = 2.0 + self.atr_24h * 20  # 高波动放大
        return (long_value + short_value) * multiplier

    # ------------------------------------------------------------------
    # ML 清算时机预测 (线性 sigmoid 模型 + 可选 LSTM 融合)
    # ------------------------------------------------------------------

    def _extract_ml_features(self) -> np.ndarray:
        """提取ML特征 (10维)

        特征顺序:
            0. LVI/100 (Liquidation Volatility Index 归一化)
            1. 杠杆/OI 比率
            2. 平均杠杆/50
            3. 资金费率×1000
            4. ATR×10
            5. RSI/100
            6. 成交量/平均成交量
            7. 多头集群数/10
            8. 空头集群数/10
            9. 极端风险标志 (0/1)
        """
        return np.array([
            self.calculate_lvi() / 100.0,
            self.total_leverage / max(self.open_interest, 1),
            self.avg_leverage / 50.0,
            self.funding_rate * 1000,
            self.atr_24h * 10,
            self.rsi / 100.0,
            self.volume / max(self.avg_volume, 1),
            len(self.long_clusters) / 10.0,
            len(self.short_clusters) / 10.0,
            1.0 if self.assess_leverage_risk() == LeverageRiskLevel.EXTREME else 0.0,
        ], dtype=np.float64)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """数值稳定的 sigmoid"""
        x = max(-500.0, min(500.0, x))
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

    def fit(self, features_array: np.ndarray, labels: np.ndarray,
            epochs: int = 50, lr: float = 0.01) -> Dict[str, float]:
        """在线训练线性 sigmoid ML 模型 (梯度下降)

        使用 sigmoid 输出 + binary cross-entropy 损失，对 (ml_weights, ml_bias)
        执行 epochs 轮全批量梯度下降。训练完成后 _model_trained 置为 True。

        Args:
            features_array: [N, ML_FEATURE_DIM] 特征矩阵
            labels: [N] 标签 (1=级联发生, 0=未发生)
            epochs: 训练轮数 (默认 50)
            lr: 学习率 (默认 0.01)

        Returns:
            {"final_loss": float, "accuracy": float}
            final_loss: 最后一轮平均 BCE 损失
            accuracy: 训练集上的二分类准确率
        """
        X = np.asarray(features_array, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[1] != self.ML_FEATURE_DIM:
            raise ValueError(
                f"features_array 形状应为 [N, {self.ML_FEATURE_DIM}], 实际为 {X.shape}"
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError("labels 长度应与 features_array 行数一致")
        if epochs < 1:
            raise ValueError("epochs 必须 >= 1")

        n = X.shape[0]
        w = self.ml_weights.copy()
        b = float(self.ml_bias)

        final_loss = 0.0
        for _ in range(epochs):
            # 前向: logits = X @ w + b
            logits = X @ w + b
            # 数值稳定的 sigmoid
            probs = np.where(
                logits >= 0,
                1.0 / (1.0 + np.exp(-logits)),
                np.exp(logits) / (1.0 + np.exp(logits)),
            )
            # BCE 损失 (clamp 避免数值问题)
            eps = 1e-12
            probs_c = np.clip(probs, eps, 1.0 - eps)
            loss = -np.mean(
                y * np.log(probs_c) + (1.0 - y) * np.log(1.0 - probs_c)
            )
            final_loss = float(loss)
            # 梯度: dL/dw = X.T @ (probs - y) / n; dL/db = mean(probs - y)
            grad_w = X.T @ (probs - y) / n
            grad_b = float(np.mean(probs - y))
            # 参数更新
            w -= lr * grad_w
            b -= lr * grad_b

        # 写回
        self.ml_weights = w
        self.ml_bias = b
        self._model_trained = True

        # 训练集准确率
        logits = X @ w + b
        preds = (logits >= 0.0).astype(np.float64)
        accuracy = float(np.mean(preds == y))

        # 同步训练 LSTM 增强器 (如果可用)
        if self.lstm_enhancer is not None and self.lstm_enhancer.is_available():
            try:
                self.lstm_enhancer.fit(features_array, labels)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LSTMEnhancer fit failed: %s", exc)

        logger.info(
            "MLLiquidationPredictor.fit: epochs=%d, n=%d, final_loss=%.6f, acc=%.4f",
            epochs, n, final_loss, accuracy,
        )
        return {"final_loss": final_loss, "accuracy": accuracy}

    def predict_30min_cascade_prob(self) -> float:
        """预测30分钟内级联概率 (线性 sigmoid 模型 + 可选 LSTM 融合)

        - 默认: 用 (ml_weights, ml_bias) 计算线性 logits → sigmoid 概率
        - 若启用 LSTM 且已训练: 双路加权融合 (线性 0.7 + LSTM 0.3)
        - 预测结果记入 _prediction_history 用于在线 EMA 准确率跟踪

        Returns:
            级联概率 [0, 1]
        """
        features = self._extract_ml_features()
        # 线性 sigmoid 模型
        score = float(np.dot(features, self.ml_weights) + self.ml_bias)
        linear_prob = self._sigmoid(score)

        # 可选 LSTM 融合
        if self.lstm_enhancer is not None and self.lstm_enhancer.is_available():
            lstm_prob = self.lstm_enhancer.predict_proba(features)
            if lstm_prob is not None:
                blend = self.ML_LSTM_BLEND
                prob = linear_prob * (1.0 - blend) + lstm_prob * blend
            else:
                prob = linear_prob
        else:
            prob = linear_prob

        prob = float(max(0.0, min(1.0, prob)))
        # 记录预测用于在线 EMA 准确率跟踪 (需后续 record_prediction_result 反馈)
        self._prediction_count += 1
        self._prediction_history.append({
            "ts": time.time(),
            "prob": prob,
            "actual": None,  # 待 record_prediction_result 填入
        })
        return prob

    def record_prediction_result(self, actual_cascade: bool) -> None:
        """记录最近一次预测的实际结果，更新在线 EMA 准确率

        Args:
            actual_cascade: True=确实发生级联, False=未发生
        """
        if len(self._prediction_history) == 0:
            return
        # 找到最近一个尚未反馈的预测
        target_idx = None
        for i in range(len(self._prediction_history) - 1, -1, -1):
            if self._prediction_history[i]["actual"] is None:
                target_idx = i
                break
        if target_idx is None:
            return
        self._prediction_history[target_idx]["actual"] = bool(actual_cascade)
        # 计算最近窗口内的准确率
        correct = 0
        total = 0
        for item in self._prediction_history:
            if item["actual"] is None:
                continue
            pred_label = item["prob"] >= 0.5
            total += 1
            if pred_label == item["actual"]:
                correct += 1
        if total > 0:
            observed = correct / total
            alpha = self.ML_ACCURACY_EMA_ALPHA
            self._accuracy_ema = float(
                (1.0 - alpha) * self._accuracy_ema + alpha * observed
            )

    # ------------------------------------------------------------------
    # Buy-the-Dip 信号
    # ------------------------------------------------------------------

    def detect_buy_dip_signal(self) -> bool:
        """检测抄底信号"""
        # 清算量骤降 (capitulation结束)
        if len(self.cascade_history) < 2:
            return False
        recent = list(self.cascade_history)[-5:]
        if len(recent) < 2:
            return False
        cascade_declining = recent[-1] < recent[-2] * 0.5
        # RSI极度超卖
        rsi_oversold = self.rsi < self.RSI_OVERSOLD
        # 成交量暴增
        volume_spike = self.volume > self.avg_volume * self.VOLUME_SPIKE_MULT
        return cascade_declining and rsi_oversold and volume_spike

    def calculate_buy_dip_target(self) -> float:
        """计算抄底目标价"""
        return self.current_price * (1.0 + self.BUY_DIP_TARGET_PCT)

    def calculate_buy_dip_stop(self) -> float:
        """计算抄底止损价"""
        return self.current_price * (1.0 - self.BUY_DIP_STOP_LOSS_PCT)

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def analyze(self) -> LiquidationSignal:
        """主分析函数"""
        signal = LiquidationSignal()

        if self.current_price <= 0 or self.open_interest <= 0:
            signal.description = "缺少市场数据"
            return signal

        lvi = self.calculate_lvi()
        risk = self.assess_leverage_risk()
        cascade_prob = self.calculate_cascade_probability()
        cascade_size = self.predict_cascade_size()
        long_value, short_value = self.get_critical_clusters()
        ml_prob = self.predict_30min_cascade_prob()

        signal.current_lvi = lvi
        signal.current_risk = risk

        # 信号决策
        if lvi > self.LVI_HIGH:
            sig_type = LiquidationSignalType.CASCADE_IMMINENT
            desc = f"级联即将发生: LVI={lvi:.1f}>50, 概率={cascade_prob*100:.0f}%"
        elif lvi > self.LVI_MODERATE:
            sig_type = LiquidationSignalType.CASCADE_WARNING
            desc = f"级联预警: LVI={lvi:.1f}, 概率={cascade_prob*100:.0f}%"
        elif self.detect_buy_dip_signal():
            sig_type = LiquidationSignalType.BUY_THE_DIP
            target = self.calculate_buy_dip_target()
            desc = (f"抄底信号: RSI={self.rsi:.0f}, 清算量骤降, "
                    f"目标={target:.0f}, 止损={self.calculate_buy_dip_stop():.0f}")
        elif risk == LeverageRiskLevel.EXTREME and self.funding_rate > self.FUNDING_RATE_EXTREME:
            sig_type = LiquidationSignalType.AVOID_LONG
            desc = f"避免做多: 多头拥挤, 资金费率={self.funding_rate*100:.3f}%"
        elif risk == LeverageRiskLevel.EXTREME and self.funding_rate < -self.FUNDING_RATE_EXTREME:
            sig_type = LiquidationSignalType.AVOID_SHORT
            desc = f"避免做空: 空头拥挤, 资金费率={self.funding_rate*100:.3f}%"
        else:
            sig_type = LiquidationSignalType.HOLD
            desc = f"风险可控: LVI={lvi:.1f}, risk={risk.value}"

        opp = LiquidationOpportunity(
            signal_type=sig_type,
            lvi=lvi,
            risk_level=risk,
            long_cluster_value=long_value,
            short_cluster_value=short_value,
            cascade_probability=cascade_prob,
            expected_cascade_size=cascade_size,
            buy_dip_target=self.calculate_buy_dip_target() if sig_type == LiquidationSignalType.BUY_THE_DIP else 0.0,
            confidence=min(1.0, cascade_prob + ml_prob * 0.3),
            description=desc,
        )
        signal.signal_type = sig_type
        signal.best_opportunity = opp
        signal.description = desc
        return signal

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "open_interest": self.open_interest,
            "total_leverage": self.total_leverage,
            "avg_leverage": self.avg_leverage,
            "funding_rate": self.funding_rate,
            "atr_24h": self.atr_24h,
            "rsi": self.rsi,
            "lvi": self.calculate_lvi(),
            "risk_level": self.assess_leverage_risk().value,
            "long_clusters": len(self.long_clusters),
            "short_clusters": len(self.short_clusters),
            "ml_30min_prob": self.predict_30min_cascade_prob(),
            "ml_accuracy_ema": float(self._accuracy_ema),
            "ml_model_trained": bool(self._model_trained),
            "ml_prediction_count": int(self._prediction_count),
            "ml_torch_enabled": self.lstm_enhancer is not None,
        }


# ============================================================================
# 可选 PyTorch 轻量增强 (默认关闭，懒加载)
# ============================================================================


class _TinyLSTM:
    """轻量 LSTM 风格网络 (懒加载 PyTorch)

    结构: Linear embedding → 1-layer LSTM → 单头自注意力 → Linear
    参数量 < 30K，CPU 推理 < 5ms

    本类不是任何论文的完整实现，仅是受 LSTM + 注意力思想启发的小型网络，
    用于与线性 sigmoid 模型双路融合。torch 不可用时 model 为 None。
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, seq_len: int = 100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self._torch = None
        self._nn = None
        self.model = None  # type: ignore[assignment]
        self._build()

    def _build(self) -> None:
        """构建网络 (torch 不可用时降级为 None)"""
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
                # 3. 单头自注意力 over T
                self.q_proj = nn.Linear(hidden, hidden)
                self.k_proj = nn.Linear(hidden, hidden)
                self.v_proj = nn.Linear(hidden, hidden)
                self.attn_scale = float(hidden) ** 0.5
                self.ln1 = nn.LayerNorm(hidden)
                # 4. 输出 (单 logit → sigmoid 概率)
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
                # 输出最后一帧
                last = ctx[:, -1, :]
                return self.output_proj(last)  # [B, 1]

        self.model = _Impl()
        self.model.eval()

    def param_count(self) -> int:
        """返回模型参数量"""
        if self._torch is None or self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))

    def predict(self, x_np: np.ndarray) -> Optional[float]:
        """前向推理，返回 logit 标量

        Args:
            x_np: [T, input_dim] numpy 数组

        Returns:
            logit 标量，torch 不可用时返回 None
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


class LSTMEnhancer:
    """PyTorch LSTM 轻量增强器 (默认关闭，懒加载)

    离线训练 + 在线推理，与线性 sigmoid 模型双路融合。
    torch 不可用时降级为 noop (predict_proba 返回 None)。

    使用：
        enhancer = LSTMEnhancer()
        enhancer.fit(features_array, labels)   # 离线训练
        prob = enhancer.predict_proba(features)  # 在线推理
    """

    def __init__(
        self,
        input_dim: int = 10,
        seq_len: int = 100,
        hidden_dim: int = 16,
        lr: float = 1e-3,
        train_epochs: int = 20,
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.train_epochs = train_epochs
        self.feature_history: deque = deque(maxlen=seq_len + 50)
        self.model: Optional[_TinyLSTM] = None
        self.optimizer = None
        self.criterion = None
        self._torch = None
        self._nn = None
        self._is_trained = False
        self._init_model()

    def _init_model(self) -> None:
        """初始化模型与优化器 (torch 不可用时降级)"""
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn = nn
        except ImportError:
            self._torch = None
            self._nn = None
            self.model = None
            logger.info("LSTMEnhancer: torch not available, fallback to noop")
            return

        self.model = _TinyLSTM(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, seq_len=self.seq_len
        )
        if self.model.model is None:
            self.model = None
            return
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.model.model.eval()
        logger.info(
            "LSTMEnhancer: initialized _TinyLSTM, params=%d",
            self.model.param_count(),
        )

    def is_available(self) -> bool:
        """torch 是否可用"""
        return self._torch is not None and self.model is not None

    def fit(self, features_array: np.ndarray, labels: np.ndarray) -> bool:
        """离线训练

        将特征矩阵构造为滑动窗口序列：用过去 seq_len 步的特征预测下一步的标签。
        若样本数不足 seq_len+5，则用零填充构造单序列。

        Returns:
            训练成功返回 True，否则 False
        """
        if not self.is_available():
            return False
        X_feat = np.asarray(features_array, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        if X_feat.ndim != 2 or X_feat.shape[1] != self.input_dim:
            return False
        if y.shape[0] != X_feat.shape[0]:
            return False

        torch = self._torch
        model = self.model.model  # type: ignore[union-attr]
        model.train()

        # 构造滑动窗口序列
        n = X_feat.shape[0]
        X_seq, Y_seq = [], []
        if n >= self.seq_len + 1:
            for i in range(n - self.seq_len):
                X_seq.append(X_feat[i:i + self.seq_len])
                Y_seq.append(y[i + self.seq_len])
        else:
            # 样本不足，零填充构造一个序列
            pad = np.zeros((self.seq_len - n, self.input_dim), dtype=np.float32)
            X_seq.append(np.vstack([pad, X_feat]))
            Y_seq.append(y[-1] if n > 0 else 0.0)

        if len(X_seq) == 0:
            model.eval()
            return False

        X_arr = np.array(X_seq, dtype=np.float32)
        Y_arr = np.array(Y_seq, dtype=np.float32)
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
            logger.warning("LSTMEnhancer fit failed: %s", exc)
            model.eval()
            return False

        model.eval()
        self._is_trained = True
        return True

    def predict_proba(self, features: np.ndarray) -> Optional[float]:
        """在线推理，返回级联概率 [0, 1]

        Args:
            features: [input_dim] 当前时刻特征向量

        Returns:
            级联概率标量，torch 不可用或未训练时返回 None
        """
        if not self.is_available():
            return None
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.shape[0] != self.input_dim:
            return None

        self.feature_history.append(feat)
        if not self._is_trained or len(self.feature_history) < 1:
            return None

        # 构造序列: 取最近 seq_len 步，不足则零填充
        hist = list(self.feature_history)
        if len(hist) >= self.seq_len:
            seq = np.array(hist[-self.seq_len:], dtype=np.float32)
        else:
            pad = np.zeros((self.seq_len - len(hist), self.input_dim), dtype=np.float32)
            seq = np.vstack([pad, np.array(hist, dtype=np.float32)])

        logit = self.model.predict(seq)  # type: ignore[union-attr]
        if logit is None:
            return None
        # sigmoid → 概率
        return float(self._sigmoid(logit))

    @staticmethod
    def _sigmoid(x: float) -> float:
        """数值稳定的 sigmoid"""
        x = max(-500.0, min(500.0, x))
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
