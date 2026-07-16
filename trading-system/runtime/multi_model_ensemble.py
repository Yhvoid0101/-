# -*- coding: utf-8 -*-
"""
多模型集成传感器 — LightGBM + Prophet + Qwen 三层架构

架构设计（来源：theneuralbase.com 2026 + systemtrade.blog 2026 + QuantMind 2026）：
  1. 速度分层：LightGBM(毫秒级) > Prophet(秒级) > Qwen(十秒级)
  2. 数据互补：LightGBM处理结构化数据，Qwen处理非结构化数据，Prophet提供趋势先验
  3. 风险分散：Qwen情绪判断出错时，LightGBM仍有硬数据作为决策依据
  4. 概率校准：所有模型输出标准化到[0,1]（theneuralbase.com 2026 最佳实践）
  5. 并行推理：三模型并行运行，总延迟=max(各模型延迟)+~1ms（非串行求和）
  6. 回退机制：任一模型故障时降级到可用模型组合（systemtrade.blog 2026）

边界原则（遵循系统架构）：
  - 三层模型是传感器，不是决策者（与TimesFM/Kronos/均线RSI地位平等）
  - 输出格式：{"direction": "long"/"short"/"neutral", "confidence": 0-1, ...}
  - 子Agent通过基因选择是否使用此信号
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hermes.multi_model")

# ============================================================================
# 常量与配置
# ============================================================================

# 模型超时阈值（秒）— 来源：1200-iron-standards 2.2节
LIGHTGBM_TIMEOUT_S = 0.1     # 毫秒级，100ms 超时
PROPHET_TIMEOUT_S = 5.0      # 秒级
QWEN_TIMEOUT_S = 15.0        # 十秒级

# 概率校准范围
PROB_MIN = 0.01
PROB_MAX = 0.99

# 信号融合权重默认值（可被 meta-learner 覆盖）
DEFAULT_WEIGHTS = {
    "lightgbm": 0.45,   # 结构化数据主力
    "prophet": 0.20,    # 趋势先验
    "qwen": 0.35,       # 情绪分析
}

# 回退权重（某模型不可用时重新归一化）
FALLBACK_WEIGHTS = {
    "lightgbm_prophet": {"lightgbm": 0.70, "prophet": 0.30},
    "lightgbm_qwen": {"lightgbm": 0.55, "qwen": 0.45},
    "prophet_qwen": {"prophet": 0.35, "qwen": 0.65},
    "lightgbm_only": {"lightgbm": 1.0},
    "prophet_only": {"prophet": 1.0},
    "qwen_only": {"qwen": 1.0},
}


# ============================================================================
# 第一层：LightGBM 实时预测（毫秒级）
# ============================================================================


class LightGBMPredictor:
    """LightGBM 实时预测层 — 处理结构化数据，毫秒级响应

    职责：
      - 输入：技术指标（RSI/MACD/ATR/布林带/均线等）
      - 输出：涨跌方向概率 [0,1]
      - 特点：CPU推理、单文件模型、支持增量训练

    来源：QuantMind 2026（LightGBM量化选股完整实践）
          QUANTA 2026（LightGBM vs LSTM 回测对比，LightGBM 310% vs LSTM 201%）
    """

    def __init__(self, model_path: str = ""):
        """
        Args:
            model_path: LightGBM 模型文件路径
        """
        self.model_path = model_path
        self._model = None
        self._booster = None
        self._model_loaded = False
        self._feature_names: List[str] = []
        self._prediction_cache: Dict[str, Tuple[float, float]] = {}
        self._cache_ttl = 1.0  # 缓存1秒

        if model_path and os.path.exists(model_path):
            self._load_model()

    def _load_model(self):
        """加载 LightGBM 模型"""
        try:
            import lightgbm as lgb
            self._booster = lgb.Booster(model_file=self.model_path)
            self._model_loaded = True
            logger.info("LightGBM model loaded: %s", self.model_path)
        except ImportError:
            logger.warning("lightgbm not installed, model unavailable")
            self._model_loaded = False
        except Exception as e:
            logger.warning("LightGBM model load failed: %s", e)
            self._model_loaded = False

    def predict(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        price_history: List[float],
    ) -> Dict[str, Any]:
        """预测下一根K线的涨跌方向

        Args:
            symbol: 交易对
            indicators: 技术指标字典（RSI/MACD/ATR/布林带等）
            price_history: 历史收盘价序列

        Returns:
            {
                "direction": "long"/"short"/"neutral",
                "confidence": 0-1,
                "predicted_return": float,  # 预期收益率
                "model_available": bool,
            }
        """
        # 缓存检查（同一K线内不重复推理）
        cache_key = f"{symbol}:{int(time.time() / self._cache_ttl)}"
        if cache_key in self._prediction_cache:
            prob, ret = self._prediction_cache[cache_key]
            return self._format_output(prob, ret, self._model_loaded)

        if self._booster is not None:
            try:
                prob, ret = self._real_predict(indicators, price_history)
            except Exception as e:
                logger.error("LightGBM predict failed: %s", e)
                return self._unavailable_output("inference_failed", "error")
        else:
            return self._unavailable_output("model_unavailable")

        self._prediction_cache[cache_key] = (prob, ret)
        # 清理过期缓存
        if len(self._prediction_cache) > 200:
            cutoff = time.time() - 10
            self._prediction_cache = {
                k: v for k, v in self._prediction_cache.items()
                if not k.endswith(str(int(time.time() / self._cache_ttl)))
            }

        return self._format_output(prob, ret, self._booster is not None)

    def _real_predict(
        self, indicators: Dict[str, Any], price_history: List[float],
    ) -> Tuple[float, float]:
        """真实 LightGBM 推理"""
        try:
            pass
            # 构建特征向量
            features = self._build_features(indicators, price_history)
            feature_array = np.array([features], dtype=np.float64)

            # 推理（毫秒级）
            prob = self._booster.predict(feature_array)[0]
            prob = float(np.clip(prob, PROB_MIN, PROB_MAX))

            # 预期收益率 = (prob - 0.5) * scale
            predicted_return = (prob - 0.5) * 0.04  # ±2% 范围

            return prob, predicted_return
        except Exception:
            raise

    def _unavailable_output(self, reason: str, status: str = "unavailable") -> Dict[str, Any]:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "predicted_return": 0.0,
            "probability": 0.5,
            "model_available": False,
            "model_name": "lightgbm",
            "status": status,
            "reason": reason,
        }

    def _mock_predict(
        self, indicators: Dict[str, Any], price_history: List[float],
    ) -> Tuple[float, float]:
        """模拟预测（基于技术指标的趋势检测）

        当 LightGBM 未安装/模型未加载时的回退方案。
        使用 RSI + 均线交叉 + 动量的简单组合模拟。
        """
        if len(price_history) < 20:
            return 0.5, 0.0

        signals = []

        # 信号1：RSI 超买超卖
        rsi = indicators.get("rsi_14", 50.0)
        if rsi < 30:
            signals.append(0.65)  # 超卖看涨
        elif rsi > 70:
            signals.append(0.35)  # 超买卖跌
        else:
            signals.append(0.50)

        # 信号2：均线交叉
        sma_20 = indicators.get("sma_20", 0.0)
        sma_50 = indicators.get("sma_50", 0.0)
        if sma_20 > sma_50:
            signals.append(0.60)  # 金叉看涨
        else:
            signals.append(0.40)  # 死叉看跌

        # 信号3：布林带位置
        bb_upper = indicators.get("bollinger_upper", 0.0)
        bb_lower = indicators.get("bollinger_lower", 0.0)
        current = price_history[-1] if price_history else 0.0
        if bb_upper > bb_lower:
            bb_pos = (current - bb_lower) / (bb_upper - bb_lower)
            # 接近下轨看涨，接近上轨看跌
            signals.append(1.0 - bb_pos)
        else:
            signals.append(0.50)

        # 信号4：短期动量
        if len(price_history) >= 5:
            momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
            # 动量 > 0 看涨
            signals.append(0.50 + min(0.20, max(-0.20, momentum * 10)))
        else:
            signals.append(0.50)

        # 加权平均
        prob = float(np.clip(np.mean(signals), PROB_MIN, PROB_MAX))
        predicted_return = (prob - 0.5) * 0.03

        return prob, predicted_return

    # 特征名称列表（与 _build_features 输出顺序严格一致）
    FEATURE_NAMES = [
        # 原始技术指标 (10)
        "rsi_14", "atr_14", "adx_14",
        "sma_20", "sma_50", "ema_12", "ema_26",
        "bollinger_upper", "bollinger_lower", "bollinger_mid",
        # 价格统计 (5)
        "current_price", "mean_price", "std_price", "return_20", "range_20",
        # 成交量特征 (4) — 来源：ericrosenfeld.ai 2026 25+技术指标最佳实践
        "volume", "volume_ma_20", "volume_ratio", "obv",
        # MACD 衍生 (3)
        "macd_dif", "macd_dea", "macd_hist",
        # 多周期收益率 (4)
        "return_1", "return_5", "return_10", "return_60",
        # 波动率特征 (4)
        "atr_pct", "bb_width", "bb_pctb", "realized_vol",
        # 动量振荡器 (3)
        "roc_10", "stoch_k", "williams_r",
        # 时间特征 (2)
        "day_of_week", "hour_of_day",
        # 滞后特征 (3)
        "lag_return_1", "lag_return_3", "lag_rsi_1",
        # 回撤特征 (2)
        "drawdown_from_high", "max_drawdown_20",
    ]

    def _build_features(
        self, indicators: Dict[str, Any], price_history: List[float],
    ) -> List[float]:
        """构建 LightGBM 特征向量（扩充版：15→40 特征）

        来源：ericrosenfeld.ai 2026（25+技术指标最佳实践）
              systemtrade.blog 2026（LightGBM 动态权重学习）
              分析报告 P0-2（特征工程不足修复）

        性能优化: 缓存 np.array(price_history[-20:]) 避免重复转换(原4次→1次),
                  向量化OBV计算替代Python循环, 合并 stoch_k/williams_r 的 min/max,
                  移除冗余 `import time as _time`(使用模块级 time)。
        """
        features = []

        # 预计算: price_history[-20:] 的 numpy 数组（原代码重复转换4次，此处缓存1次）
        has_20 = len(price_history) >= 20
        prices_20 = np.array(price_history[-20:]) if has_20 else None

        # === 1. 原始技术指标 (10) ===
        feature_keys = [
            "rsi_14", "atr_14", "adx_14",
            "sma_20", "sma_50", "ema_12", "ema_26",
            "bollinger_upper", "bollinger_lower", "bollinger_mid",
        ]
        for key in feature_keys:
            val = indicators.get(key)
            features.append(float(val) if val is not None else 50.0 if "rsi" in key else 0.0)

        # === 2. 价格统计 (5) ===
        if has_20:
            features.extend([
                float(prices_20[-1]),                           # 当前价格
                float(np.mean(prices_20)),                       # 均价
                float(np.std(prices_20)),                        # 波动率
                float((prices_20[-1] - prices_20[0]) / prices_20[0]) if prices_20[0] != 0 else 0.0,  # 20期收益率
                float(np.max(prices_20) - np.min(prices_20)),       # 振幅
            ])
        else:
            features.extend([0.0] * 5)

        # === 3. 成交量特征 (4) ===
        # 来源：ericrosenfeld.ai 2026 — 量价关系是量化核心
        volume = indicators.get("volume", 0.0)
        volume_history = indicators.get("volume_history", [])
        if volume_history and len(volume_history) >= 20:
            vol_arr = np.array(volume_history[-20:])
            volume_ma_20 = float(np.mean(vol_arr))
            volume_ratio = float(volume / volume_ma_20) if volume_ma_20 > 0 else 1.0
            # OBV (On Balance Volume) — 向量化计算替代 Python 循环
            price_arr = np.array(price_history[-len(vol_arr):]) if len(price_history) >= len(vol_arr) else np.array(price_history)
            n = min(len(vol_arr), len(price_arr))
            if n > 1:
                price_diff = np.diff(price_arr[:n])
                vol_shifted = np.asarray(vol_arr[1:n], dtype=np.float64)
                obv = float(np.where(price_diff > 0, vol_shifted, np.where(price_diff < 0, -vol_shifted, 0.0)).sum())
            else:
                obv = 0.0
        else:
            volume_ma_20 = float(volume) if volume > 0 else 1.0
            volume_ratio = 1.0
            obv = 0.0
        features.extend([float(volume), volume_ma_20, volume_ratio, obv])

        # === 4. MACD 衍生特征 (3) ===
        ema_12 = indicators.get("ema_12", 0.0)
        ema_26 = indicators.get("ema_26", 0.0)
        macd_dif = float(ema_12) - float(ema_26)  # DIF = EMA12 - EMA26
        macd_dea = float(indicators.get("macd_dea", macd_dif * 0.9))  # DEA = EMA9(DIF)
        macd_hist = macd_dif - macd_dea  # MACD柱 = DIF - DEA
        features.extend([macd_dif, macd_dea, macd_hist])

        # === 5. 多周期收益率 (4) ===
        if len(price_history) >= 60:
            ph = price_history
            ret_1 = (ph[-1] - ph[-2]) / ph[-2] if ph[-2] != 0 else 0.0
            ret_5 = (ph[-1] - ph[-6]) / ph[-6] if len(ph) >= 6 and ph[-6] != 0 else 0.0
            ret_10 = (ph[-1] - ph[-11]) / ph[-11] if len(ph) >= 11 and ph[-11] != 0 else 0.0
            ret_60 = (ph[-1] - ph[-61]) / ph[-61] if len(ph) >= 61 and ph[-61] != 0 else 0.0
        else:
            ret_1 = ret_5 = ret_10 = ret_60 = 0.0
        features.extend([float(ret_1), float(ret_5), float(ret_10), float(ret_60)])

        # === 6. 波动率特征 (4) ===
        atr_14 = float(indicators.get("atr_14", 0.0))
        current_price = float(price_history[-1]) if price_history else 1.0
        atr_pct = atr_14 / current_price if current_price > 0 else 0.0
        bb_upper = float(indicators.get("bollinger_upper", 0.0))
        bb_lower = float(indicators.get("bollinger_lower", 0.0))
        bb_mid = float(indicators.get("bollinger_mid", 0.0))
        bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0.0
        bb_pctb = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        # 使用缓存的 prices_20 计算 realized_vol（原代码重复 np.array 转换）
        if len(price_history) >= 21 and prices_20 is not None:
            realized_vol = float(np.std(np.diff(np.log(prices_20)))) * np.sqrt(20)
        else:
            realized_vol = 0.0
        features.extend([float(atr_pct), float(bb_width), float(bb_pctb), float(realized_vol)])

        # === 7. 动量振荡器 (3) ===
        roc_10 = (price_history[-1] - price_history[-11]) / price_history[-11] * 100 if len(price_history) >= 11 and price_history[-11] != 0 else 0.0
        # 随机指标 %K + Williams %R — 合并 min/max 计算（原代码分别计算2次）
        if len(price_history) >= 14:
            last_14 = np.array(price_history[-14:])
            low_14 = float(np.min(last_14))
            high_14 = float(np.max(last_14))
            range_14 = high_14 - low_14
            stoch_k = (price_history[-1] - low_14) / range_14 * 100 if range_14 > 0 else 50.0
            williams_r = (high_14 - price_history[-1]) / range_14 * -100 if range_14 > 0 else -50.0
        else:
            stoch_k = 50.0
            williams_r = -50.0
        features.extend([float(roc_10), float(stoch_k), float(williams_r)])

        # === 8. 时间特征 (2) ===
        # 来源：ericrosenfeld.ai 2026 — 周期性对预测至关重要
        # 性能优化: 移除冗余 `import time as _time`，使用模块级 `time`（已在文件顶部导入）
        now = time.time()
        day_of_week = int(time.strftime("%w", time.localtime(now)))  # 0=Sunday
        hour_of_day = int(time.strftime("%H", time.localtime(now)))
        features.extend([float(day_of_week), float(hour_of_day)])

        # === 9. 滞后特征 (3) ===
        lag_ret_1 = (price_history[-2] - price_history[-3]) / price_history[-3] if len(price_history) >= 3 and price_history[-3] != 0 else 0.0
        lag_ret_3 = (price_history[-4] - price_history[-7]) / price_history[-7] if len(price_history) >= 7 and price_history[-7] != 0 else 0.0
        lag_rsi_1 = float(indicators.get("rsi_14", 50.0))  # 简化：用当前RSI作为滞后近似
        features.extend([float(lag_ret_1), float(lag_ret_3), lag_rsi_1])

        # === 10. 回撤特征 (2) ===
        if has_20 and prices_20 is not None:
            running_max = np.maximum.accumulate(prices_20)
            drawdown = (prices_20[-1] - running_max[-1]) / running_max[-1] if running_max[-1] > 0 else 0.0
            max_dd = float(np.min((prices_20 - running_max) / running_max))
        else:
            drawdown = 0.0
            max_dd = 0.0
        features.extend([float(drawdown), float(max_dd)])

        return features

    def _format_output(self, prob: float, predicted_return: float, available: bool) -> Dict[str, Any]:
        """格式化输出为标准传感器格式"""
        if prob > 0.55:
            direction = "long"
        elif prob < 0.45:
            direction = "short"
        else:
            direction = "neutral"

        confidence = abs(prob - 0.5) * 2  # 0-1

        return {
            "direction": direction,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "predicted_return": float(predicted_return),
            "probability": float(prob),
            "model_available": available,
            "model_name": "lightgbm",
            "status": "available" if available else "unavailable",
        }

    def train_incremental(
        self, features: np.ndarray, labels: np.ndarray,
    ) -> bool:
        """增量训练（盘后微调）

        Args:
            features: 特征矩阵 (n_samples, n_features)
            labels: 标签 (n_samples,) — 1=涨, 0=跌

        Returns:
            训练是否成功
        """
        try:
            import lightgbm as lgb
            if self._booster is None:
                # 首次训练
                train_data = lgb.Dataset(features, label=labels)
                params = {
                    "objective": "binary",
                    "metric": "auc",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                    "verbose": -1,
                }
                self._booster = lgb.train(params, train_data, num_boost_round=100)
                logger.info("LightGBM initial training done: %d samples", len(labels))
            else:
                # 增量训练
                train_data = lgb.Dataset(features, label=labels)
                self._booster = lgb.train(
                    {}, train_data,
                    num_boost_round=20,
                    init_model=self._booster,
                    keep_training_booster=True,
                )
                logger.info("LightGBM incremental training done: %d samples", len(labels))
            return True
        except ImportError:
            logger.warning("lightgbm not installed, cannot train")
            return False
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)
            return False


# ============================================================================
# 第二层：Prophet 趋势分析（秒级）
# ============================================================================


class ProphetTrendAnalyzer:
    """Prophet 趋势分析层 — 提供趋势方向和季节性先验

    职责：
      - 输入：历史价格序列（至少30个点）
      - 输出：趋势方向、趋势强度、季节性分量、预测区间
      - 特点：自动检测变点、节假日效应、多季节性分解

    来源：Meta Prophet（Taylor & Letham 2018）
          flowt.fr 2026（Prophet vs LightGBM 时间序列对比）
    """

    def __init__(self, model_cache_dir: str = ""):
        """
        Args:
            model_cache_dir: Prophet 模型缓存目录（持久化已训练模型，避免每次重训）
        """
        # 延迟导入优化: prophet 包导入耗时 ~1s (含 cmdstanpy/plotly/pandas 链式导入)。
        # 原实现在 __init__ 中同步导入, 导致实例化阻塞 1013ms。
        # 现改为首次 analyze() 调用时按需导入 (_ensure_prophet_checked)。
        self._prophet_available = None  # None=未检测, True/False=已检测（延迟导入）
        self._last_forecast: Dict[str, Any] = {}
        self._forecast_ttl = 60.0  # 60秒缓存
        self._model_cache_dir = model_cache_dir or os.path.expanduser("~/hermes_models/prophet_cache")
        self._cached_models: Dict[str, Any] = {}  # symbol -> Prophet model
        self._model_timestamps: Dict[str, float] = {}  # symbol -> last train timestamp
        self._model_ttl = 3600.0  # 模型缓存1小时（避免每次重训）

    def _ensure_prophet_checked(self) -> bool:
        """首次调用时检测 prophet 可用性（延迟导入，避免 __init__ 阻塞 ~1s）

        prophet 包导入会触发 cmdstanpy/plotly/pandas 链式导入，冷启动耗时约 1s。
        移到首次 analyze() 时按需导入，使 __init__ 从 1013ms 降至 ~0.01ms。
        幂等：检测结果缓存到 _prophet_available，后续调用直接返回。
        """
        if self._prophet_available is not None:
            return self._prophet_available
        try:
            from prophet import Prophet  # noqa: F401
            self._prophet_available = True
            # 创建缓存目录
            os.makedirs(self._model_cache_dir, exist_ok=True)
            logger.info("Prophet available, cache dir: %s", self._model_cache_dir)
        except ImportError:
            logger.warning("prophet not installed, model unavailable")
            self._prophet_available = False
        except Exception as e:
            logger.warning("Prophet init failed: %s", e)
            self._prophet_available = False
        return self._prophet_available

    def analyze(
        self,
        symbol: str,
        price_history: List[float],
        timestamps: Optional[List[float]] = None,
        horizon: int = 10,
    ) -> Dict[str, Any]:
        """分析价格趋势

        Args:
            symbol: 交易对
            price_history: 历史收盘价序列（至少30个点）
            timestamps: 对应时间戳（可选，默认用序号）
            horizon: 预测步长

        Returns:
            {
                "direction": "long"/"short"/"neutral",
                "confidence": 0-1,
                "trend_strength": 0-1,
                "seasonality": float,       # 季节性分量强度
                "forecast_price": float,    # 预测价格
                "forecast_lower": float,    # 预测下界
                "forecast_upper": float,    # 预测上界
                "model_available": bool,
            }
        """
        if len(price_history) < 30:
            return self._default_output(price_history)

        # 缓存检查
        cache_key = f"{symbol}:{int(time.time() / self._forecast_ttl)}"
        if cache_key in self._last_forecast:
            return self._last_forecast[cache_key]

        if self._ensure_prophet_checked():
            try:
                result = self._real_prophet(symbol, price_history, timestamps, horizon)
            except Exception as e:
                logger.error("Prophet analysis failed: %s", e)
                result = self._unavailable_output("inference_failed", "error")
        else:
            result = self._unavailable_output("model_unavailable")

        self._last_forecast[cache_key] = result
        # 清理过期缓存
        if len(self._last_forecast) > 50:
            self._last_forecast.clear()

        return result

    def _real_prophet(
        self,
        symbol: str,
        price_history: List[float],
        timestamps: Optional[List[float]],
        horizon: int,
    ) -> Dict[str, Any]:
        """真实 Prophet 推理（P1改进：参数调优+模型持久化）

        改进点：
          1. changepoint_prior_scale 品种自适应（高波动→大值）
          2. changepoint_range=0.9（覆盖更近的变点）
          3. 模型持久化（训练后缓存，1小时内不重训）
          4. 加密货币关闭 weekly_seasonality（24/7交易无周内效应）
        """
        try:
            from prophet import Prophet
            import pandas as pd

            # 构建 Prophet 所需的 DataFrame
            if timestamps:
                dates = pd.to_datetime(timestamps, unit="s")
            else:
                dates = pd.date_range(end="2026-01-01", periods=len(price_history), freq="D")

            df = pd.DataFrame({"ds": dates, "y": price_history})

            # === P1改进1: 品种自适应参数 ===
            # 来源：Prophet 官方文档 + 分析报告 P1
            # 金融数据噪声大，0.05 默认值会导致过拟合伪变点
            prices_arr = np.array(price_history)
            volatility = float(np.std(np.diff(np.log(prices_arr)))) if len(prices_arr) > 1 else 0.02
            # 高波动品种（如加密货币）用更大的 changepoint_prior_scale
            if volatility > 0.05:
                cps = 0.1  # 高波动：0.1（更刚性，避免过拟合噪声）
                weekly_seas = False  # 加密货币24/7交易，无周内效应
            elif volatility > 0.02:
                cps = 0.05  # 中等波动：0.05
                weekly_seas = True
            else:
                cps = 0.05  # 低波动：0.05
                weekly_seas = True

            # === P1改进2: 模型持久化 ===
            # 检查是否有缓存的模型（1小时内不重训）
            model = self._get_cached_model(symbol)
            if model is None:
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=10.0,
                    changepoint_range=0.9,  # 覆盖到更近的数据（默认0.8只覆盖80%）
                    daily_seasonality=False,
                    weekly_seasonality=weekly_seas,
                    yearly_seasonality=False,
                )
                model.fit(df)
                self._cache_model(symbol, model)
                logger.info("Prophet model trained and cached for %s (cps=%.2f, vol=%.4f)", symbol, cps, volatility)

            # 预测
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            # 提取趋势信息
            trend_values = forecast["trend"].values
            last_trend = trend_values[-1]
            first_trend = trend_values[0]
            trend_slope = (last_trend - first_trend) / len(trend_values)

            # 趋势方向
            if trend_slope > 0.001:
                direction = "long"
            elif trend_slope < -0.001:
                direction = "short"
            else:
                direction = "neutral"

            # 趋势强度（归一化）
            trend_strength = float(np.clip(abs(trend_slope) * 1000, 0.0, 1.0))

            # 季节性分量
            if "weekly" in forecast.columns:
                seasonality = float(np.std(forecast["weekly"].values))
            else:
                seasonality = 0.0

            # 预测区间
            yhat = forecast["yhat"].values[-1]
            yhat_lower = forecast["yhat_lower"].values[-1]
            yhat_upper = forecast["yhat_upper"].values[-1]

            confidence = trend_strength

            return {
                "direction": direction,
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "trend_strength": trend_strength,
                "seasonality": seasonality,
                "forecast_price": float(yhat),
                "forecast_lower": float(yhat_lower),
                "forecast_upper": float(yhat_upper),
                "model_available": True,
                "model_name": "prophet",
            }

        except Exception as e:
            logger.error("Prophet analysis failed: %s", e)
            return self._unavailable_output("inference_failed", "error")

    def _unavailable_output(self, reason: str, status: str = "unavailable") -> Dict[str, Any]:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "trend_strength": 0.0,
            "seasonality": 0.0,
            "forecast_price": None,
            "forecast_lower": None,
            "forecast_upper": None,
            "model_available": False,
            "model_name": "prophet",
            "status": status,
            "reason": reason,
        }

    def _mock_prophet(
        self, symbol: str, price_history: List[float], horizon: int,
    ) -> Dict[str, Any]:
        """模拟趋势分析（基于线性回归 + 季节性分解）

        当 Prophet 未安装时的回退方案。
        """
        prices = np.array(price_history[-60:] if len(price_history) >= 60 else price_history)
        n = len(prices)

        # 线性趋势检测
        x = np.arange(n)
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # 趋势方向
        avg_price = np.mean(prices)
        relative_slope = slope / avg_price if avg_price > 0 else 0.0

        if relative_slope > 0.001:
            direction = "long"
        elif relative_slope < -0.001:
            direction = "short"
        else:
            direction = "neutral"

        # 趋势强度
        trend_strength = float(np.clip(abs(relative_slope) * 500, 0.0, 1.0))

        # 简单季节性检测（7日周期）
        if n >= 14:
            seasonal_values = prices - np.polyval(coeffs, x)
            seasonality = float(np.std(seasonal_values))
        else:
            seasonality = 0.0

        # 预测
        forecast_price = float(np.polyval(coeffs, n + horizon))
        forecast_std = float(np.std(prices - np.polyval(coeffs, x))) if n > 2 else avg_price * 0.02
        forecast_lower = forecast_price - 1.96 * forecast_std
        forecast_upper = forecast_price + 1.96 * forecast_std

        confidence = trend_strength

        return {
            "direction": direction,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "trend_strength": trend_strength,
            "seasonality": seasonality,
            "forecast_price": forecast_price,
            "forecast_lower": float(forecast_lower),
            "forecast_upper": float(forecast_upper),
            "model_available": False,
            "model_name": "prophet_mock",
        }

    def _get_cached_model(self, symbol: str):
        """获取缓存的 Prophet 模型（P1改进：模型持久化）

        优先从内存缓存读取，其次从磁盘读取。
        模型缓存1小时，超时后重新训练。
        """
        now = time.time()
        # 内存缓存检查
        if symbol in self._cached_models:
            train_time = self._model_timestamps.get(symbol, 0)
            if now - train_time < self._model_ttl:
                return self._cached_models[symbol]
            # 超时，清除
            del self._cached_models[symbol]
            self._model_timestamps.pop(symbol, None)

        # 磁盘缓存检查
        cache_file = os.path.join(self._model_cache_dir, f"{symbol}_prophet.pkl")
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if now - mtime < self._model_ttl:
                try:
                    import pickle
                    with open(cache_file, "rb") as f:
                        model = pickle.load(f)
                    self._cached_models[symbol] = model
                    self._model_timestamps[symbol] = mtime
                    logger.info("Prophet model loaded from disk cache: %s", symbol)
                    return model
                except Exception as e:
                    logger.warning("Failed to load Prophet cache: %s", e)

        return None

    def _cache_model(self, symbol: str, model):
        """缓存 Prophet 模型到内存和磁盘"""
        self._cached_models[symbol] = model
        self._model_timestamps[symbol] = time.time()

        # 磁盘持久化
        cache_file = os.path.join(self._model_cache_dir, f"{symbol}_prophet.pkl")
        try:
            import pickle
            with open(cache_file, "wb") as f:
                pickle.dump(model, f)
            logger.debug("Prophet model cached to disk: %s", cache_file)
        except Exception as e:
            logger.warning("Failed to cache Prophet model: %s", e)

    def _default_output(self, price_history: List[float]) -> Dict[str, Any]:
        """数据不足时的默认输出"""
        last_price = price_history[-1] if price_history else 0.0
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "trend_strength": 0.0,
            "seasonality": 0.0,
            "forecast_price": last_price,
            "forecast_lower": last_price,
            "forecast_upper": last_price,
            "model_available": False,
            "model_name": "prophet",
            "status": "unavailable",
        }


# ============================================================================
# 第三层：Qwen 情绪分析（十秒级）
# ============================================================================


class QwenSentimentAnalyzer:
    """Qwen 情绪分析层 — 处理非结构化文本数据

    职责：
      - 输入：新闻标题、社交媒体文本、财报摘要
      - 输出：情绪分数 [-1,1]、情绪标签、置信度
      - 特点：LLM语义理解、缓存机制、超时降级

    来源：theneuralbase.com 2026（LightGBM + LLM 集成最佳实践）
          搜索结果关键建议：LLM输出需标准化到[0,1]再送入meta-learner

    风险控制：
      - Qwen情绪信号只是一个输入特征，不是最终决策者
      - LightGBM基于硬数据的信号作为安全网
      - 超时/出错时自动降级，不影响实时交易
    """

    def __init__(self, api_router_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            api_router_config: API路由配置（含Qwen模型地址和参数）
        """
        self._config = api_router_config or {}
        self._ollama_url = self._config.get("ollama_url", "http://localhost:11434")
        self._model_name = self._config.get("model", "qwen2.5:0.5b")
        self._qwen_available = False
        self._sentiment_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_ttl = 300.0  # 5分钟缓存

        # 检查 Qwen 是否可用
        self._check_availability()

    def _check_availability(self):
        """检查 Qwen/Ollama 是否可用"""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self._ollama_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    self._qwen_available = True
                    logger.info("Qwen/Ollama available at %s", self._ollama_url)
        except Exception:
            logger.warning("Qwen/Ollama not available, using mock sentiment")
            self._qwen_available = False

    def analyze(
        self,
        symbol: str,
        news_texts: List[str],
        social_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """分析市场情绪

        Args:
            symbol: 交易对
            news_texts: 新闻标题列表
            social_texts: 社交媒体文本列表（可选）

        Returns:
            {
                "direction": "long"/"short"/"neutral",
                "confidence": 0-1,
                "sentiment_score": -1 to 1,  # 情绪分数
                "sentiment_label": "bullish"/"bearish"/"neutral",
                "key_factors": List[str],     # 关键影响因素
                "model_available": bool,
            }
        """
        all_texts = news_texts + (social_texts or [])
        if not all_texts:
            return self._default_output()

        # 缓存检查（文本内容未变时不重复调用）
        cache_key = self._hash_texts(symbol, all_texts)
        cached = self._sentiment_cache.get(cache_key)
        if cached and (time.time() - cached[1] < self._cache_ttl):
            return cached[0]

        if self._qwen_available:
            result = self._real_qwen_analyze(symbol, all_texts)
        else:
            result = {
                "direction": "neutral",
                "confidence": 0.0,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "key_factors": [],
                "model_available": False,
                "model_name": "qwen",
                "status": "unavailable",
            }

        self._sentiment_cache[cache_key] = (result, time.time())
        # 清理过期缓存
        if len(self._sentiment_cache) > 100:
            cutoff = time.time() - self._cache_ttl
            self._sentiment_cache = {
                k: v for k, v in self._sentiment_cache.items() if v[1] > cutoff
            }

        return result

    def _real_qwen_analyze(self, symbol: str, texts: List[str]) -> Dict[str, Any]:
        """真实 Qwen 情绪分析"""
        try:
            import urllib.request
            import json as json_mod

            # 构建提示词（改进版：显式约束 label-score 一致性 + Few-shot 示例）
            # 来源：theneuralbase.com 2026 + jisem-journal.com 2026 LLM金融情绪分析最佳实践
            combined_text = " | ".join(texts[:10])  # 最多取10条
            prompt = (
                f"你是专业的金融市场情绪分析师。分析以下关于{symbol}的市场文本。\n"
                f"文本：{combined_text}\n\n"
                f"请严格遵循以下规则：\n"
                f"1. sentiment_score: -1到1的浮点数（-1=极度看跌, 0=中性, 1=极度看涨）\n"
                f"2. sentiment_label: 必须与score一致\n"
                f"   - score > 0.3 → bullish\n"
                f"   - score < -0.3 → bearish\n"
                f"   - -0.3 ≤ score ≤ 0.3 → neutral\n"
                f"3. confidence: 0到1的浮点数（你对判断的把握程度）\n"
                f"4. key_factors: 影响判断的关键词列表\n\n"
                f"示例：\n"
                f'  文本"BTC暴涨突破新高" → {{"sentiment_score": 0.8, "sentiment_label": "bullish", "confidence": 0.9, "key_factors": ["暴涨", "突破新高"]}}\n'
                f'  文本"暴跌利空崩盘" → {{"sentiment_score": -0.9, "sentiment_label": "bearish", "confidence": 0.85, "key_factors": ["暴跌", "利空", "崩盘"]}}\n\n'
                f"只输出JSON，不要其他内容。"
            )

            payload = json_mod.dumps({
                "model": self._model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # 强制结构化输出（Ollama JSON mode）
                "options": {"temperature": 0.2, "num_predict": 256},  # 降低温度提高一致性
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=QWEN_TIMEOUT_S) as resp:
                data = json_mod.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "")

            # 解析 Qwen 输出的 JSON
            result = self._parse_qwen_response(response_text)
            return result

        except Exception as e:
            logger.error("Qwen analysis failed: %s", e)
            return {"model_available": False, "status": "error", "error": str(e)}

    def _parse_qwen_response(self, response: str) -> Dict[str, Any]:
        """解析 Qwen 的 JSON 响应（含 label-score 一致性校验）

        修复：之前 label 和 score 无交叉校验，导致 bullish+score=-1 的矛盾输出
        来源：分析报告 P0-1 + jisem-journal.com 2026
        """
        import re

        # 尝试提取 JSON（支持嵌套 key_factors 数组）
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                sentiment_score = float(np.clip(data.get("sentiment_score", 0.0), -1.0, 1.0))
                sentiment_label = data.get("sentiment_label", "neutral")
                confidence = float(np.clip(data.get("confidence", 0.5), 0.0, 1.0))
                key_factors = data.get("key_factors", [])

                # === label-score 一致性校验（核心修复）===
                # 以 score 为准修正 label，消除矛盾
                expected_label = (
                    "bullish" if sentiment_score > 0.3
                    else "bearish" if sentiment_score < -0.3
                    else "neutral"
                )
                if sentiment_label != expected_label:
                    logger.warning(
                        "Qwen label-score inconsistent: label=%s score=%.4f → corrected to %s",
                        sentiment_label, sentiment_score, expected_label,
                    )
                    sentiment_label = expected_label
                    # 矛盾时降低置信度（模型输出不可靠）
                    confidence *= 0.7

                # 概率校准到 [0,1]（theneuralbase.com 2026 最佳实践）
                prob = (sentiment_score + 1.0) / 2.0
                prob = float(np.clip(prob, PROB_MIN, PROB_MAX))

                if prob > 0.55:
                    direction = "long"
                elif prob < 0.45:
                    direction = "short"
                else:
                    direction = "neutral"

                return {
                    "direction": direction,
                    "confidence": confidence,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                    "key_factors": key_factors if isinstance(key_factors, list) else [],
                    "probability": prob,
                    "model_available": True,
                    "model_name": "qwen",
                }
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning("Qwen response parse failed: %s", e)

        # 解析失败，返回中性
        return self._default_output(available=True)

    def _mock_qwen_analyze(self, symbol: str, texts: List[str]) -> Dict[str, Any]:
        """模拟情绪分析（基于关键词匹配）

        当 Qwen 不可用时的回退方案。
        使用简单的金融情绪词典进行匹配。
        """
        # 情绪词典
        bullish_words = [
            "上涨", "利好", "突破", "牛市", "增持", "买入", "强势", "新高",
            "surge", "bullish", "breakout", "rally", "upgrade", "buy",
        ]
        bearish_words = [
            "下跌", "利空", "暴跌", "熊市", "减持", "卖出", "弱势", "新低",
            "crash", "bearish", "breakdown", "sell-off", "downgrade", "sell",
        ]

        combined = " ".join(texts).lower()
        bullish_count = sum(1 for w in bullish_words if w in combined)
        bearish_count = sum(1 for w in bearish_words if w in combined)

        total = bullish_count + bearish_count
        if total == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
            confidence = 0.3
        else:
            sentiment_score = (bullish_count - bearish_count) / total
            if sentiment_score > 0.2:
                sentiment_label = "bullish"
            elif sentiment_score < -0.2:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"
            confidence = float(np.clip(total / 10.0, 0.3, 0.8))

        # 概率校准
        prob = (sentiment_score + 1.0) / 2.0
        prob = float(np.clip(prob, PROB_MIN, PROB_MAX))

        if prob > 0.55:
            direction = "long"
        elif prob < 0.45:
            direction = "short"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "confidence": confidence,
            "sentiment_score": float(sentiment_score),
            "sentiment_label": sentiment_label,
            "key_factors": [],
            "probability": prob,
            "model_available": False,
            "model_name": "qwen_mock",
        }

    def _default_output(self, available: bool = False) -> Dict[str, Any]:
        """无文本输入时的默认输出"""
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "key_factors": [],
            "probability": 0.5,
            "model_available": available,
            "model_name": "qwen",
        }

    def _hash_texts(self, symbol: str, texts: List[str]) -> str:
        """生成文本内容的缓存键"""
        import hashlib
        combined = symbol + "|" + "|".join(texts)
        return hashlib.md5(combined.encode("utf-8")).hexdigest()


# ============================================================================
# 模型性能监控 — 追踪各模型预测准确率
# ============================================================================


class ModelPerformanceMonitor:
    """模型性能监控器 — 追踪各模型近期预测准确率

    来源：CSDN 2026（AI量化交易闭环实战）— 模型性能监控与漂移检测
          arxiv.org 2026（Adaptive Alpha Weighting with PPO）— 动态权重调整

    功能：
      1. 记录每个模型的预测方向和实际方向
      2. 计算滑动窗口准确率（EMA 加权）
      3. 检测模型性能衰减（概念漂移）
      4. 为动态权重调整提供数据支撑
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑动窗口大小（记录最近 N 次预测）
        """
        self.window_size = window_size
        # 每个模型的预测记录: {model_name: deque([(predicted, actual, timestamp), ...])}
        self._predictions: Dict[str, deque] = {
            "lightgbm": deque(maxlen=window_size),
            "prophet": deque(maxlen=window_size),
            "qwen": deque(maxlen=window_size),
            "ensemble": deque(maxlen=window_size),
        }
        # 模型间预测相关性记录
        self._correlation_history: List[Dict[str, float]] = []

    def record_prediction(
        self,
        model_name: str,
        predicted_direction: str,
        predicted_probability: float,
        timestamp: float,
    ) -> None:
        """记录模型预测（实际结果未到时先记录预测）"""
        if model_name not in self._predictions:
            self._predictions[model_name] = deque(maxlen=self.window_size)
        self._predictions[model_name].append({
            "predicted": predicted_direction,
            "probability": predicted_probability,
            "actual": None,  # 等待实际结果
            "timestamp": timestamp,
        })

    def update_actual_result(
        self,
        timestamp: float,
        actual_direction: str,
        lookback_seconds: float = 60.0,
    ) -> None:
        """更新实际结果（回填到匹配的预测记录）

        Args:
            timestamp: 当前时间戳
            actual_direction: 实际涨跌方向
            lookback_seconds: 回看多少秒内的预测（匹配未填充的记录）
        """
        cutoff = timestamp - lookback_seconds
        for model_name, records in self._predictions.items():
            for rec in records:
                if rec["actual"] is None and rec["timestamp"] >= cutoff:
                    rec["actual"] = actual_direction

    def get_accuracy(self, model_name: str) -> float:
        """获取模型近期准确率（已验证的记录中）"""
        records = self._predictions.get(model_name, deque())
        verified = [r for r in records if r["actual"] is not None]
        if not verified:
            return 0.5  # 无数据时返回中性
        correct = sum(1 for r in verified if r["predicted"] == r["actual"])
        return correct / len(verified)

    def get_ema_accuracy(self, model_name: str, alpha: float = 0.3) -> float:
        """获取 EMA 加权准确率（近期表现权重更高）

        Args:
            model_name: 模型名称
            alpha: EMA 衰减因子（0-1，越小越平滑）
        """
        records = self._predictions.get(model_name, deque())
        verified = [r for r in records if r["actual"] is not None]
        if not verified:
            return 0.5
        ema = 0.5  # 初始值
        for r in verified:
            correct = 1.0 if r["predicted"] == r["actual"] else 0.0
            ema = alpha * correct + (1 - alpha) * ema
        return ema

    def detect_performance_drift(self, model_name: str, threshold: float = 0.15) -> Dict[str, Any]:
        """检测模型性能衰减（概念漂移）

        比较前半段和后半段的准确率，差异超过阈值则告警

        Returns:
            {"drifted": bool, "early_acc": float, "recent_acc": float, "delta": float}
        """
        records = list(self._predictions.get(model_name, deque()))
        verified = [r for r in records if r["actual"] is not None]
        if len(verified) < 20:
            return {"drifted": False, "reason": "insufficient_data"}

        mid = len(verified) // 2
        early = verified[:mid]
        recent = verified[mid:]

        early_acc = sum(1 for r in early if r["predicted"] == r["actual"]) / len(early)
        recent_acc = sum(1 for r in recent if r["predicted"] == r["actual"]) / len(recent)
        delta = recent_acc - early_acc

        return {
            "drifted": delta < -threshold,
            "early_acc": early_acc,
            "recent_acc": recent_acc,
            "delta": delta,
        }

    def get_model_correlation(self) -> Dict[str, float]:
        """计算模型间预测相关性（集成多样性度量）

        高相关性（>0.9）意味着模型冗余，集成无意义
        """
        models = ["lightgbm", "prophet", "qwen"]
        correlations = {}
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                recs1 = [r for r in self._predictions.get(m1, deque()) if r["actual"] is not None]
                recs2 = [r for r in self._predictions.get(m2, deque()) if r["actual"] is not None]
                if len(recs1) < 10 or len(recs2) < 10:
                    correlations[f"{m1}_{m2}"] = 0.5
                    continue
                # 取共同时间戳的记录
                min_len = min(len(recs1), len(recs2))
                dirs1 = [r["predicted"] for r in recs1[-min_len:]]
                dirs2 = [r["predicted"] for r in recs2[-min_len:]]
                # 简单相关：方向一致比例
                agree = sum(1 for d1, d2 in zip(dirs1, dirs2) if d1 == d2) / min_len
                correlations[f"{m1}_{m2}"] = agree
        return correlations

    def get_stats(self) -> Dict[str, Any]:
        """获取所有模型性能统计"""
        stats = {}
        for model_name in self._predictions:
            records = self._predictions[model_name]
            verified = [r for r in records if r["actual"] is not None]
            stats[model_name] = {
                "total_predictions": len(records),
                "verified_predictions": len(verified),
                "accuracy": self.get_accuracy(model_name),
                "ema_accuracy": self.get_ema_accuracy(model_name),
                "drift": self.detect_performance_drift(model_name),
            }
        stats["correlations"] = self.get_model_correlation()
        return stats


# ============================================================================
# 动态权重管理器 — 基于近期表现自动调整模型权重
# ============================================================================


class DynamicWeightManager:
    """动态权重管理器 — 根据模型近期表现自动调整权重

    来源：systemtrade.blog 2026（LightGBM 动态权重学习）
          arxiv.org 2026（Adaptive Alpha Weighting with PPO）
          CSDN 2026（AI量化交易闭环实战）— Stacking/Blending 集成

    策略：
      1. 基础权重：DEFAULT_WEIGHTS（0.45/0.20/0.35）
      2. 表现加权：softmax(EMA准确率 / temperature)
      3. 市场状态条件权重：趋势市场提高 LightGBM/Prophet，震荡市场提高 Qwen
      4. 性能衰减检测：衰减模型自动降权
      5. Meta-learner：已训练时用逻辑回归替代加权平均
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        performance_monitor: Optional[ModelPerformanceMonitor] = None,
        temperature: float = 0.15,
    ):
        """
        Args:
            base_weights: 基础权重
            performance_monitor: 性能监控器
            temperature: softmax 温度（越小权重差异越大）
        """
        self.base_weights = base_weights or dict(DEFAULT_WEIGHTS)
        self.monitor = performance_monitor or ModelPerformanceMonitor()
        self.temperature = temperature
        self.quarantine: Dict[str, Dict[str, Any]] = {}
        self._market_regime = "unknown"  # trend / range / volatile / unknown
        self._regime_weights = {
            "trend": {"lightgbm": 0.55, "prophet": 0.30, "qwen": 0.15},
            "range": {"lightgbm": 0.30, "prophet": 0.15, "qwen": 0.55},
            "volatile": {"lightgbm": 0.35, "prophet": 0.20, "qwen": 0.45},
            "unknown": None,  # 使用基础权重
        }

    def set_market_regime(self, regime: str):
        """设置当前市场状态（由 regime_detection 模块提供）

        Args:
            regime: "trend" / "range" / "volatile" / "unknown"

        P0-2 修复: 仅在 regime 变化时记录日志, 避免每 tick 重复输出
                  "Market regime set to: volatile" 导致日志爆炸
        根因: regime_detection 模块每 tick 调用本方法, 即使 regime 未变化也记录 INFO 日志
        影响: 500 轮进化日志膨胀至 2.2MB+, 关键告警被淹没
        来源: /tmp/intraday_500_v235_final.log 大量重复 "Market regime set to: volatile"
        """
        if regime in self._regime_weights:
            if regime != self._market_regime:
                self._market_regime = regime
                logger.info("Market regime set to: %s", regime)

    def get_weights(self, active_models: List[str]) -> Dict[str, float]:
        """获取动态权重

        权重计算流程：
          1. 确定基础权重（市场状态条件权重 or 默认权重）
          2. 用 EMA 准确率调整（表现好的模型加权）
          3. 检测性能衰减（衰减模型降权）
          4. 归一化到 [0,1] 且和为 1
        """
        if not active_models:
            return {}

        eligible_models = []
        for model in active_models:
            drift = self.monitor.detect_performance_drift(model)
            if drift.get("drifted", False):
                first_quarantine = model not in self.quarantine
                self.quarantine[model] = {
                    "reason": "performance_drift",
                    "drift": dict(drift),
                    "timestamp": time.time(),
                }
                if first_quarantine:
                    logger.error("Model %s quarantined due to performance drift", model)
            else:
                eligible_models.append(model)
        active_models = eligible_models
        if not active_models:
            return {}

        # 步骤1：基础权重
        regime_weights = self._regime_weights.get(self._market_regime)
        if regime_weights:
            weights = {m: regime_weights.get(m, 0.0) for m in active_models}
        else:
            weights = {m: self.base_weights.get(m, 0.0) for m in active_models}

        # 步骤2：用 EMA 准确率调整
        for model in active_models:
            ema_acc = self.monitor.get_ema_accuracy(model)
            # 准确率偏离 0.5 越多，权重调整越大
            adjustment = (ema_acc - 0.5) * 2  # -1 to 1
            weights[model] *= (1.0 + adjustment * 0.5)  # ±50% 调整
            weights[model] = max(0.05, weights[model])  # 最低权重保底

        # 步骤4：归一化
        total = sum(weights.values())
        if total > 0:
            weights = {m: w / total for m, w in weights.items()}
        else:
            equal = 1.0 / len(active_models)
            weights = {m: equal for m in active_models}

        return weights

    def get_stats(self) -> Dict[str, Any]:
        """获取动态权重管理器状态"""
        return {
            "market_regime": self._market_regime,
            "base_weights": self.base_weights,
            "temperature": self.temperature,
            "performance": self.monitor.get_stats(),
            "quarantine": self.quarantine,
        }


# ============================================================================
# 信号融合层 — Meta-Learner 加权决策
# ============================================================================


class SignalFusionLayer:
    """信号融合层 — 将三个模型的输出融合为统一交易信号

    来源：theneuralbase.com 2026（Combining LightGBM and LLM predictions）
          CSDN 2026（AI量化交易闭环实战）— Stacking/Blending 集成
          关键建议：所有模型输出必须校准到[0,1]后再送入meta-learner

    融合策略（P0-3 修复：激活 meta-learner 空壳）：
      1. 概率校准：所有模型输出标准化到 [0,1]
      2. 动态权重：基于 EMA 准确率 + 市场状态条件权重
      3. Meta-learner：已训练时用逻辑回归替代加权平均（消除空壳）
      4. 方向一致性检查：三模型方向不一致时降低置信度
      5. 回退机制：某模型不可用时重新归一化权重
      6. 反馈闭环：记录预测用于后续训练
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        performance_monitor: Optional[ModelPerformanceMonitor] = None,
    ):
        """
        Args:
            weights: 模型权重 {"lightgbm": x, "prophet": y, "qwen": z}
            performance_monitor: 外部传入的性能监控器（共享实例）
        """
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self._meta_learner = None
        self._meta_learner_trained = False
        self._sklearn_checked = False  # 延迟导入标记
        self._fusion_history: List[Dict[str, Any]] = []
        self._max_history = 1000

        # 性能监控器和动态权重管理器（P0-3/P0-4 新增）
        self.monitor = performance_monitor or ModelPerformanceMonitor()
        self.dynamic_weights = DynamicWeightManager(
            base_weights=self.weights,
            performance_monitor=self.monitor,
        )

        # 延迟导入优化: sklearn.linear_model 导入耗时 ~1.3s (含 scipy/scipy.stats 链式导入)。
        # 原实现在 __init__ 中同步导入, 导致实例化阻塞 1318ms。
        # 现改为首次使用 meta-learner 时按需导入 (_ensure_meta_learner)。

    def _ensure_meta_learner(self):
        """首次使用时导入 sklearn 并初始化 meta-learner（延迟导入，避免 __init__ 阻塞 ~1.3s）

        scikit-learn 导入会触发 scipy/scipy.stats 链式导入，冷启动耗时约 1.3s。
        移到首次 fuse()/train_meta_learner() 时按需导入，使 __init__ 从 1318ms 降至 ~0.01ms。
        幂等：导入检测结果缓存到 _sklearn_checked，后续调用直接返回已初始化的 _meta_learner。
        """
        if self._sklearn_checked:
            return self._meta_learner
        self._sklearn_checked = True
        try:
            from sklearn.linear_model import LogisticRegression
            self._meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            logger.info("Meta-learner initialized (LogisticRegression)")
        except ImportError:
            logger.warning("scikit-learn not installed, using weighted average fusion")
        return self._meta_learner

    def fuse(
        self,
        lightgbm_signal: Dict[str, Any],
        prophet_signal: Dict[str, Any],
        qwen_signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """融合三个模型的信号

        Args:
            lightgbm_signal: LightGBM 预测输出
            prophet_signal: Prophet 趋势分析输出
            qwen_signal: Qwen 情绪分析输出

        Returns:
            {
                "direction": "long"/"short"/"neutral",
                "confidence": 0-1,
                "fused_probability": float,    # 融合后概率
                "model_agreement": float,      # 模型一致性 0-1
                "active_models": List[str],    # 参与融合的模型
                "weights_used": Dict[str, float],
                "meta_learner_used": bool,     # 是否使用了 meta-learner
            }
        """
        # 收集可用模型
        signals = {}
        if lightgbm_signal.get("model_available") is True:
            signals["lightgbm"] = lightgbm_signal
        if prophet_signal.get("model_available") is True:
            signals["prophet"] = prophet_signal
        if qwen_signal.get("model_available") is True:
            signals["qwen"] = qwen_signal

        if not signals:
            return self._default_fused_output()

        # 计算各模型概率（校准到 [0,1]）
        probs = {}
        for name, sig in signals.items():
            prob = sig.get("probability")
            if prob is None:
                direction = sig.get("direction", "neutral")
                confidence = sig.get("confidence", 0.5)
                if direction == "long":
                    prob = 0.5 + confidence * 0.5
                elif direction == "short":
                    prob = 0.5 - confidence * 0.5
                else:
                    prob = 0.5
            probs[name] = float(np.clip(prob, PROB_MIN, PROB_MAX))

        # === 获取动态权重（P0-3 修复：不再使用静态权重）===
        active_models = list(signals.keys())
        weights = self.dynamic_weights.get_weights(active_models)

        # === Meta-learner 融合（P0-3 修复：激活空壳）===
        meta_learner_used = False
        meta_learner = self._ensure_meta_learner()
        if meta_learner is not None and self._meta_learner_trained:
            try:
                # 用 meta-learner 预测融合概率
                feature_vector = np.array([[probs.get(m, 0.5) for m in ["lightgbm", "prophet", "qwen"]]])
                fused_prob = float(meta_learner.predict_proba(feature_vector)[0, 1])
                fused_prob = float(np.clip(fused_prob, PROB_MIN, PROB_MAX))
                meta_learner_used = True
            except Exception as e:
                logger.warning("Meta-learner predict failed: %s, fallback to weighted average", e)
                fused_prob = sum(probs[name] * weights[name] for name in signals)
                fused_prob = float(np.clip(fused_prob, PROB_MIN, PROB_MAX))
        else:
            # 加权平均融合
            fused_prob = sum(probs[name] * weights[name] for name in signals)
            fused_prob = float(np.clip(fused_prob, PROB_MIN, PROB_MAX))

        # 方向判定
        if fused_prob > 0.55:
            direction = "long"
        elif fused_prob < 0.45:
            direction = "short"
        else:
            direction = "neutral"

        # 模型一致性检查
        directions = [sig.get("direction", "neutral") for sig in signals.values()]
        agreement = self._compute_agreement(directions)

        # 置信度 = 融合概率偏离0.5的程度 * 模型一致性
        base_confidence = abs(fused_prob - 0.5) * 2
        confidence = float(np.clip(base_confidence * agreement, 0.0, 1.0))

        result = {
            "direction": direction,
            "confidence": confidence,
            "fused_probability": fused_prob,
            "model_agreement": agreement,
            "active_models": active_models,
            "weights_used": weights,
            "meta_learner_used": meta_learner_used,
            "individual_signals": {
                name: {
                    "direction": sig.get("direction", "neutral"),
                    "confidence": sig.get("confidence", 0.0),
                    "probability": probs[name],
                }
                for name, sig in signals.items()
            },
        }

        # 记录融合历史 + 性能监控（反馈闭环）
        self._record_fusion(result)
        timestamp = time.time()
        for name in active_models:
            self.monitor.record_prediction(
                model_name=name,
                predicted_direction=result["individual_signals"][name]["direction"],
                predicted_probability=probs[name],
                timestamp=timestamp,
            )
        self.monitor.record_prediction(
            model_name="ensemble",
            predicted_direction=direction,
            predicted_probability=fused_prob,
            timestamp=timestamp,
        )

        # 自动训练 meta-learner（每 100 次融合且未训练时）
        if (not self._meta_learner_trained and
                len(self._fusion_history) >= 50 and
                self._meta_learner is not None):
            self._try_auto_train_meta_learner()

        return result

    def _get_active_weights(self, active_models: List[str]) -> Dict[str, float]:
        """根据可用模型确定权重（回退机制）"""
        if len(active_models) == 3:
            return dict(self.weights)

        # 查找回退权重
        key = "_".join(sorted(active_models))
        if key in FALLBACK_WEIGHTS:
            return dict(FALLBACK_WEIGHTS[key])

        # 动态归一化
        total = sum(self.weights.get(m, 0.0) for m in active_models)
        if total > 0:
            return {m: self.weights.get(m, 0.0) / total for m in active_models}

        # 全部相等
        equal = 1.0 / len(active_models)
        return {m: equal for m in active_models}

    def _compute_agreement(self, directions: List[str]) -> float:
        """计算模型方向一致性

        Returns:
            0.0-1.0，1.0 = 完全一致，0.0 = 完全矛盾
        """
        if not directions:
            return 0.0

        long_count = directions.count("long")
        short_count = directions.count("short")
        neutral_count = directions.count("neutral")

        total = len(directions)
        max_count = max(long_count, short_count, neutral_count)

        # 一致性 = 最大方向占比 * (1 - 中性稀释)
        agreement = max_count / total
        if neutral_count > 0:
            agreement *= (1 - neutral_count / (total * 2))

        return float(np.clip(agreement, 0.0, 1.0))

    def _record_fusion(self, result: Dict[str, Any]):
        """记录融合历史（用于 meta-learner 训练）"""
        self._fusion_history.append({
            "timestamp": time.time(),
            "result": result,
        })
        if len(self._fusion_history) > self._max_history:
            self._fusion_history = self._fusion_history[-self._max_history:]

    def _default_fused_output(self) -> Dict[str, Any]:
        """无可用模型时的默认输出"""
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "fused_probability": 0.5,
            "model_agreement": 0.0,
            "active_models": [],
            "weights_used": {},
            "meta_learner_used": False,
            "tradable": False,
            "status": "unavailable",
            "reason": "no_available_models",
        }

    def _try_auto_train_meta_learner(self):
        """自动训练 meta-learner（使用已验证的预测历史）

        从性能监控器中提取已验证的预测记录，按时间戳对齐3个模型的概率，
        构建正确的多特征训练数据：
          - 特征矩阵 X: (n_samples, 3) — 每行是 [lightgbm_prob, prophet_prob, qwen_prob]
          - 标签向量 y: (n_samples,) — 1.0=long, 0.0=short/neutral

        对齐策略：三模型在同一 fuse() 调用中记录，时间戳完全相同
        （fuse() 中 timestamp = time.time() 只调用一次，传给3个模型）。
        按时间戳分组，每组3个模型概率组成一个特征向量。
        使用精确匹配（==）而非容差匹配，因为同一 fuse() 调用的
        3个模型记录使用同一个 float 值，保证完全相等。
        不同 fuse() 调用即使间隔很短（如测试环境<1ms），时间戳也不同，
        不会误合并。

        来源：systemtrade.blog 2026（动态权重学习）+ 修复"纸面meta-learner"缺陷
        """
        meta_learner = self._ensure_meta_learner()
        if meta_learner is None:
            return
        try:
            models = ["lightgbm", "prophet", "qwen"]

            # 步骤1：收集每个模型的已验证记录，按时间戳索引
            # timestamp -> {model_name: probability}
            timestamp_aligned: Dict[float, Dict[str, Tuple[float, str]]] = {}

            for model_name in models:
                records = self.monitor._predictions.get(model_name, deque())
                for rec in records:
                    if rec["actual"] is not None:
                        ts = rec["timestamp"]
                        # 精确匹配：同一 fuse() 调用的3个模型时间戳完全相同
                        # （同一个 float 值传给所有 record_prediction 调用）
                        # 不同 fuse() 调用即使间隔<1ms，时间戳也不同
                        if ts not in timestamp_aligned:
                            timestamp_aligned[ts] = {}
                        timestamp_aligned[ts][model_name] = (
                            rec["probability"],
                            rec["actual"],
                        )

            # 步骤2：只保留3个模型都有记录的时间戳（完整样本）
            features = []
            labels = []
            for ts, model_data in timestamp_aligned.items():
                if len(model_data) == 3:  # 3个模型都有记录
                    # 特征向量：[lightgbm_prob, prophet_prob, qwen_prob]
                    feat = [model_data[m][0] for m in models]
                    # 标签：取任一模型的actual（同一时间戳下应该一致）
                    label_val = list(model_data.values())[0][1]
                    features.append(feat)
                    labels.append(1.0 if label_val == "long" else 0.0)

            if len(features) < 20:
                logger.debug("Not enough aligned data for meta-learner: %d samples (need 20)", len(features))
                return

            # 步骤3：构建特征矩阵 (n_samples, 3) 并训练
            X = np.array(features)  # shape: (n, 3)
            y = np.array(labels)    # shape: (n,)

            meta_learner.fit(X, y)
            self._meta_learner_trained = True
            logger.info("Meta-learner auto-trained with %d aligned samples (3 features each)", len(features))
        except Exception as e:
            logger.warning("Auto-train meta-learner failed: %s", e)
            import traceback
            logger.debug(traceback.format_exc())

    def train_meta_learner(self, features: np.ndarray, labels: np.ndarray) -> bool:
        """训练 meta-learner（逻辑回归）

        Args:
            features: 各模型概率组成的特征矩阵 (n_samples, 3)
            labels: 实际涨跌标签 (n_samples,)

        Returns:
            训练是否成功
        """
        meta_learner = self._ensure_meta_learner()
        if meta_learner is None:
            return False

        try:
            meta_learner.fit(features, labels)
            self._meta_learner_trained = True
            logger.info("Meta-learner trained: %d samples", len(labels))
            return True
        except Exception as e:
            logger.error("Meta-learner training failed: %s", e)
            return False


# ============================================================================
# 系统入口 — 多模型集成传感器
# ============================================================================


class MultiModelEnsembleSystem:
    """多模型集成传感器系统 — 三层架构统一入口

    架构：
      LightGBM(毫秒级) ──┐
      Prophet(秒级)   ──┼──> SignalFusionLayer ──> 统一交易信号
      Qwen(十秒级)     ──┘

    特点：
      1. 并行推理：三模型并行运行，总延迟 = max(各模型延迟) + ~1ms
      2. 回退机制：任一模型故障时降级到可用模型组合
      3. 传感器地位：输出 {direction, confidence} 格式，与TimesFM/Kronos地位平等
    """

    def __init__(
        self,
        lightgbm_model_path: str = "",
        api_router_config: Optional[Dict[str, Any]] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            lightgbm_model_path: LightGBM 模型文件路径
            api_router_config: Qwen API 路由配置
            fusion_weights: 信号融合权重
        """
        self.lightgbm = LightGBMPredictor(model_path=lightgbm_model_path)
        self.prophet = ProphetTrendAnalyzer()
        self.qwen = QwenSentimentAnalyzer(api_router_config=api_router_config)
        # 共享性能监控器实例（P0-3/P0-4 新增）
        self.performance_monitor = ModelPerformanceMonitor()
        self.fusion = SignalFusionLayer(
            weights=fusion_weights,
            performance_monitor=self.performance_monitor,
        )

        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="model")
        self._last_signal: Dict[str, Any] = {}
        self._signal_ttl = 1.0  # 1秒缓存

        logger.info(
            "MultiModelEnsembleSystem initialized: "
            "LightGBM=%s, Prophet=%s, Qwen=%s",
            self.lightgbm._booster is not None,
            bool(self.prophet._prophet_available),
            self.qwen._qwen_available,
        )

    def get_ensemble_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        price_history: List[float],
        timestamps: Optional[List[float]] = None,
        news_texts: Optional[List[str]] = None,
        social_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """获取多模型融合信号

        这是系统的主要入口，返回融合后的统一交易信号。

        Args:
            symbol: 交易对
            indicators: 技术指标字典
            price_history: 历史收盘价序列
            timestamps: 时间戳序列（Prophet用）
            news_texts: 新闻文本列表（Qwen用）
            social_texts: 社交媒体文本列表（Qwen用）

        Returns:
            融合后的信号 + 各模型独立信号
        """
        # 缓存检查
        cache_key = f"{symbol}:{int(time.time() / self._signal_ttl)}"
        if cache_key in self._last_signal:
            return self._last_signal[cache_key]

        # 并行调用三个模型
        futures = {}

        # LightGBM（毫秒级）
        if "lightgbm" in self.fusion.dynamic_weights.quarantine:
            results = {"lightgbm": {
                "model_available": False,
                "status": "quarantined",
                "model_name": "lightgbm",
            }}
        else:
            futures["lightgbm"] = self._executor.submit(
                self.lightgbm.predict, symbol, indicators, price_history,
            )

        # Prophet（秒级）
        if "prophet" not in self.fusion.dynamic_weights.quarantine:
            futures["prophet"] = self._executor.submit(
                self.prophet.analyze, symbol, price_history, timestamps,
            )

        # Qwen（十秒级）
        if "qwen" not in self.fusion.dynamic_weights.quarantine:
            futures["qwen"] = self._executor.submit(
                self.qwen.analyze, symbol, news_texts or [], social_texts,
            )

        # 收集结果（带超时）
        results = locals().get("results", {})
        timeouts = {"lightgbm": LIGHTGBM_TIMEOUT_S, "prophet": PROPHET_TIMEOUT_S, "qwen": QWEN_TIMEOUT_S}

        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=timeouts[name])
            except FuturesTimeoutError:
                logger.warning("%s timed out (%.1fs), skipping", name, timeouts[name])
                results[name] = self._default_signal(name)
            except Exception as e:
                logger.warning("%s failed: %s, skipping", name, e)
                results[name] = self._default_signal(name)

        # 信号融合
        fused = self.fusion.fuse(
            results.get("lightgbm", {}),
            results.get("prophet", {}),
            results.get("qwen", {}),
        )

        # 组合输出
        output = {
            "ensemble": fused,
            "lightgbm": results.get("lightgbm", {}),
            "prophet": results.get("prophet", {}),
            "qwen": results.get("qwen", {}),
        }

        self._last_signal[cache_key] = output
        if len(self._last_signal) > 100:
            self._last_signal.clear()

        return output

    def _default_signal(self, model_name: str) -> Dict[str, Any]:
        """模型失败时的默认信号"""
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "model_available": False,
            "model_name": model_name,
        }

    def health_check(self) -> Dict[str, Any]:
        """系统健康检查

        Returns:
            各模型可用性状态 + 性能监控 + 动态权重
        """
        return {
            "lightgbm": {
                "available": self.lightgbm._booster is not None,
                "model_path": self.lightgbm.model_path,
            },
            "prophet": {
                "available": bool(self.prophet._prophet_available),
            },
            "qwen": {
                "available": self.qwen._qwen_available,
                "model": self.qwen._model_name,
                "url": self.qwen._ollama_url,
            },
            "fusion": {
                "weights": self.fusion.weights,
                "history_size": len(self.fusion._fusion_history),
                "meta_learner_trained": self.fusion._meta_learner_trained,
                "dynamic_weights": self.fusion.dynamic_weights.get_stats(),
            },
            "performance": self.performance_monitor.get_stats(),
        }

    def set_market_regime(self, regime: str):
        """设置当前市场状态（由 regime_detection 模块调用）

        Args:
            regime: "trend" / "range" / "volatile" / "unknown"
        """
        self.fusion.dynamic_weights.set_market_regime(regime)

    def update_actual_result(self, actual_direction: str, lookback_seconds: float = 60.0):
        """反馈实际结果（反馈闭环）

        在每根K线收盘后调用，将实际涨跌方向回填到预测记录中，
        用于计算模型准确率和自动训练 meta-learner。

        Args:
            actual_direction: "long" / "short" / "neutral"
            lookback_seconds: 回看多少秒内的预测
        """
        self.performance_monitor.update_actual_result(
            timestamp=time.time(),
            actual_direction=actual_direction,
            lookback_seconds=lookback_seconds,
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取模型性能统计"""
        return self.performance_monitor.get_stats()

    def shutdown(self):
        """关闭系统，释放资源"""
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("MultiModelEnsembleSystem shutdown")


# ============================================================================
# 异常输入检测 — 数据质量门禁
# ============================================================================


class InputValidator:
    """异常输入检测器 — 在模型推理前验证数据质量

    来源：CSDN 2026（AI量化交易闭环实战）— 数据治理与质量控制
          ericrosenfeld.ai 2026 — 内置数据质量检查

    检查项：
      1. 价格为零/负值/NaN
      2. 价格突变（单日>20%）
      3. 数据缺失（长度不足）
      4. 成交量为零
      5. 技术指标异常值（RSI>100 或 <0）
    """

    @staticmethod
    def validate_price_history(price_history: List[float]) -> Dict[str, Any]:
        """验证价格历史数据质量

        Returns:
            {"valid": bool, "errors": List[str], "warnings": List[str]}
        """
        errors = []
        warnings = []

        if not price_history:
            errors.append("price_history is empty")
            return {"valid": False, "errors": errors, "warnings": warnings}

        if len(price_history) < 20:
            errors.append(f"price_history too short: {len(price_history)} < 20 (minimum 20 required for model inference)")

        for i, price in enumerate(price_history):
            if price is None or (isinstance(price, float) and np.isnan(price)):
                errors.append(f"NaN price at index {i}")
            elif price <= 0:
                errors.append(f"non-positive price at index {i}: {price}")

        # 价格突变检测（单日变动>20%）
        for i in range(1, len(price_history)):
            if price_history[i-1] > 0:
                change = abs(price_history[i] - price_history[i-1]) / price_history[i-1]
                if change > 0.20:
                    warnings.append(f"price jump at index {i}: {change:.1%}")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    @staticmethod
    def validate_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
        """验证技术指标数据质量"""
        errors = []
        warnings = []

        rsi = indicators.get("rsi_14")
        if rsi is not None:
            if isinstance(rsi, float) and np.isnan(rsi):
                errors.append(f"RSI is NaN")
            elif rsi < 0 or rsi > 100:
                errors.append(f"RSI out of range: {rsi}")

        for key in ["sma_20", "sma_50", "ema_12", "ema_26"]:
            val = indicators.get(key)
            if val is not None:
                if isinstance(val, float) and np.isnan(val):
                    errors.append(f"{key} is NaN")
                elif val < 0:
                    warnings.append(f"{key} is negative: {val}")

        bb_upper = indicators.get("bollinger_upper", 0)
        bb_lower = indicators.get("bollinger_lower", 0)
        if isinstance(bb_upper, float) and np.isnan(bb_upper):
            errors.append("bollinger_upper is NaN")
        elif isinstance(bb_lower, float) and np.isnan(bb_lower):
            errors.append("bollinger_lower is NaN")
        elif bb_upper < bb_lower:
            errors.append("bollinger_upper < bollinger_lower")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    @staticmethod
    def validate_all(
        indicators: Dict[str, Any],
        price_history: List[float],
        news_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """验证所有输入数据"""
        price_result = InputValidator.validate_price_history(price_history)
        indicator_result = InputValidator.validate_indicators(indicators)

        all_errors = price_result["errors"] + indicator_result["errors"]
        all_warnings = price_result["warnings"] + indicator_result["warnings"]

        if news_texts is not None and not news_texts:
            all_warnings.append("news_texts is empty (Qwen will return neutral)")

        return {
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "warnings": all_warnings,
        }


# ============================================================================
# 回测验证模块 — 验证融合信号的历史表现
# ============================================================================


class BacktestValidator:
    """回测验证器 — 验证多模型融合信号的历史表现

    来源：ericrosenfeld.ai 2026（Walk-forward backtesting + Monte Carlo）
          CSDN 2026（AI量化交易闭环实战）— 回测验证体系

    功能：
      1. Walk-Forward 回测：滚动训练/测试窗口验证泛化能力
      2. 信号准确率统计：方向命中率、置信度校准
      3. 简单策略回测：基于融合信号的多空策略表现
      4. 蒙特卡洛置换检验：验证信号是否显著优于随机
    """

    def __init__(self, ensemble_system: "MultiModelEnsembleSystem"):
        """
        Args:
            ensemble_system: 多模型集成系统实例
        """
        self.system = ensemble_system
        self._backtest_results: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        symbol: str,
        price_history: List[float],
        indicators_history: List[Dict[str, Any]],
        news_texts_per_step: Optional[List[List[str]]] = None,
        window_size: int = 60,
        step_size: int = 1,
    ) -> Dict[str, Any]:
        """运行 Walk-Forward 回测

        Args:
            symbol: 交易对
            price_history: 完整价格历史
            indicators_history: 每步的技术指标
            news_texts_per_step: 每步的新闻文本
            window_size: 回看窗口大小
            step_size: 步长

        Returns:
            {
                "total_signals": int,
                "correct_signals": int,
                "accuracy": float,
                "long_accuracy": float,
                "short_accuracy": float,
                "avg_confidence": float,
                "calibration": float,  # 置信度校准（高置信度时准确率是否更高）
                "equity_curve": List[float],  # 简单策略净值曲线
                "max_drawdown": float,
                "sharpe_ratio": float,
            }
        """
        n = len(price_history)
        if n < window_size + 10:
            return {"error": "insufficient data for backtest"}

        signals = []
        actual_directions = []
        confidences = []

        for i in range(window_size, n - 1, step_size):
            window_prices = price_history[max(0, i - window_size):i + 1]
            indicators = indicators_history[i] if i < len(indicators_history) else {}
            news = news_texts_per_step[i] if news_texts_per_step and i < len(news_texts_per_step) else []

            # 输入验证
            validation = InputValidator.validate_all(indicators, window_prices, news)
            if not validation["valid"]:
                continue

            # 获取融合信号
            signal = self.system.get_ensemble_signal(
                symbol=symbol,
                indicators=indicators,
                price_history=window_prices,
                news_texts=news,
            )
            ensemble = signal.get("ensemble", {})
            direction = ensemble.get("direction", "neutral")
            confidence = ensemble.get("confidence", 0.0)

            # 实际方向
            actual = "long" if price_history[i + 1] > price_history[i] else "short"

            signals.append(direction)
            actual_directions.append(actual)
            confidences.append(confidence)

        if not signals:
            return {"error": "no valid signals generated"}

        # 统计
        correct = sum(1 for s, a in zip(signals, actual_directions) if s == a)
        total = len(signals)
        accuracy = correct / total if total > 0 else 0.0

        long_signals = [(s, a) for s, a in zip(signals, actual_directions) if s == "long"]
        short_signals = [(s, a) for s, a in zip(signals, actual_directions) if s == "short"]
        long_acc = sum(1 for s, a in long_signals if s == a) / len(long_signals) if long_signals else 0.0
        short_acc = sum(1 for s, a in short_signals if s == a) / len(short_signals) if short_signals else 0.0

        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # 置信度校准：高置信度（>0.5）时的准确率 vs 低置信度
        high_conf_mask = np.array(confidences) > 0.5
        if high_conf_mask.any():
            high_conf_correct = sum(
                1 for i, (s, a) in enumerate(zip(signals, actual_directions))
                if high_conf_mask[i] and s == a
            )
            high_conf_total = int(high_conf_mask.sum())
            calibration = high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0
        else:
            calibration = 0.0

        # 简单策略净值曲线
        equity = [1.0]
        for i, (s, a) in enumerate(zip(signals, actual_directions)):
            if s == "long":
                ret = 0.01 if a == "long" else -0.01
            elif s == "short":
                ret = 0.01 if a == "short" else -0.01
            else:
                ret = 0.0  # neutral 不交易
            equity.append(equity[-1] * (1 + ret))

        # 最大回撤
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio（简化：假设无风险利率=0）
        returns = np.diff(equity)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        result = {
            "total_signals": total,
            "correct_signals": correct,
            "accuracy": accuracy,
            "long_accuracy": long_acc,
            "short_accuracy": short_acc,
            "avg_confidence": avg_conf,
            "calibration": calibration,
            "equity_curve": equity,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
        }

        self._backtest_results.append(result)
        return result

    def monte_carlo_test(
        self,
        signals: List[str],
        actual_directions: List[str],
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """蒙特卡洛置换检验：验证信号是否显著优于随机

        Args:
            signals: 信号列表
            actual_directions: 实际方向列表
            n_simulations: 模拟次数

        Returns:
            {
                "actual_accuracy": float,
                "random_mean": float,
                "random_std": float,
                "p_value": float,
                "significant": bool,  # p < 0.05
            }
        """
        actual_acc = sum(1 for s, a in zip(signals, actual_directions) if s == a) / len(signals)

        # 随机置换检验
        random_accs = []
        for _ in range(n_simulations):
            shuffled = list(signals)
            np.random.shuffle(shuffled)
            rand_acc = sum(1 for s, a in zip(shuffled, actual_directions) if s == a) / len(signals)
            random_accs.append(rand_acc)

        random_mean = float(np.mean(random_accs))
        random_std = float(np.std(random_accs))
        p_value = float(np.mean(np.array(random_accs) >= actual_acc))

        return {
            "actual_accuracy": actual_acc,
            "random_mean": random_mean,
            "random_std": random_std,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
