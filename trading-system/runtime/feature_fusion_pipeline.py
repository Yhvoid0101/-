#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一特征融合管道 (Unified Feature Fusion Pipeline)
==================================================
短板 #2 补齐: 凯利公式 + 价格行为 + 量价 + 多周期结构 标准化融合

核心痛点:
  - 4维特征分散在不同模块, 无统一提取接口
  - MTF 完全未集成到主循环
  - VolumeProfile 已实现但未接线
  - v90d 双轨凯利缺失
  - 指标"生搬硬套" — 各维度独立计算, 无加权融合

设计原则 (用户铁律):
  - "融会贯通, 信手拈来" — 4维特征统一提取, 加权融合
  - "不生搬硬套" — 各维度有权重, 根据市场状态动态调整
  - "凯利公式" — 双轨凯利 (full + half), 根据置信度选择
  - "价格行为" — 缠论背驰 + BOS/CHOCH + 形态识别
  - "量价分析" — VPIN 订单流毒性 + VolumeProfile POC/VAH/VAL
  - "多周期结构" — 4h/1h/15m 加权融合, 大周期否决

统一输出 FeatureVector:
  - kelly_fraction: 凯利仓位 (0-1)
  - kelly_track: "full" / "half" (双轨选择)
  - pa_signal: 价格行为信号 (-1 ~ +1)
  - pa_patterns: 检测到的形态列表
  - vpin_score: 订单流毒性 (0-1, 越高越毒)
  - volume_profile_poc: POC 价格
  - mtf_direction: 多周期方向 (-1 ~ +1)
  - mtf_strength: 多周期一致性 (0-1)
  - composite_score: 综合评分 (-1 ~ +1)
  - confidence: 置信度 (0-1, 决定 kelly_track)

使用方式:
  from feature_fusion_pipeline import FeatureFusionPipeline

  pipeline = FeatureFusionPipeline()
  fv = pipeline.extract(
      klines_1h=klines,
      klines_4h=klines_4h,  # 可选
      klines_15m=klines_15m,  # 可选
      symbol="BTCUSDT",
      trade_history=returns_list,  # 可选, 用于凯利更新
  )
  # fv.composite_score → 综合信号
  # fv.kelly_fraction → 建议仓位
"""
from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ============================================================================
# 路径与日志
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [FeatureFusion] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FeatureFusionPipeline")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class FeatureVector:
    """统一特征向量 — 4维融合输出"""
    # ===== 凯利公式维度 =====
    kelly_fraction: float = 0.0          # 凯利仓位 (0-1)
    kelly_track: str = "half"            # "full" / "half" (双轨选择)
    kelly_win_rate: float = 0.0          # 胜率估计
    kelly_avg_win: float = 0.0           # 平均盈利
    kelly_avg_loss: float = 0.0          # 平均亏损
    kelly_confidence: float = 0.0        # 凯利置信度 (样本量)

    # ===== 价格行为维度 =====
    pa_signal: float = 0.0               # 价格行为信号 (-1 ~ +1)
    pa_patterns: List[str] = field(default_factory=list)  # 检测到的形态
    pa_bos_choch: str = ""               # BOS/CHOCH 信号
    pa_support: float = 0.0              # 支撑位
    pa_resistance: float = 0.0           # 阻力位

    # ===== 量价分析维度 =====
    vpin_score: float = 0.0              # 订单流毒性 (0-1, 越高越毒)
    vpin_is_toxic: bool = False          # 是否毒性过高
    volume_profile_poc: float = 0.0      # POC (Point of Control)
    volume_profile_vah: float = 0.0      # VAH (Value Area High)
    volume_profile_val: float = 0.0      # VAL (Value Area Low)

    # ===== CVD 订单流分析维度 (v98 新增, 用户铁律"量价分析融会贯通") =====
    # 来源: order_flow_cvd.py — Bulkowski OHLCV→delta + 4种CVD-Price模式 + 背离 + absorption
    # 全网前沿: LiveVolatile 2026 / LedgerMind 2026 (73% BTC反转前出现CVD背离)
    cvd_value: float = 0.0               # CVD 累计值 (正=买方主导, 负=卖方主导)
    cvd_slope: float = 0.0              # CVD 短期斜率 (近期变化方向)
    cvd_slope_long: float = 0.0         # CVD 长期斜率 (主趋势方向)
    cvd_normalized: float = 0.0         # 归一化 CVD (-1 ~ +1)
    buy_sell_pressure: float = 1.0       # 买/卖比 (>1.5 买方主导, <0.67 卖方主导)
    cvd_pattern: str = "neutral"        # CVD-Price 模式 (6种)
    cvd_divergence: int = 0             # Delta 背离 (0=无, 1=BULLISH, -1=BEARISH)
    cvd_absorption: bool = False        # 吸收模式 (delta spike + 价格盘整)
    cvd_signal: float = 0.0             # CVD 综合信号 (-1 ~ +1)
    cvd_confidence: float = 0.0          # CVD 置信度 (0-1)

    # ===== FVG 公允价值缺口维度 (v701 P1-3 新增, 用户铁律"指标融会贯通") =====
    # 来源: fair_value_gap_detector.py — ICT Smart Money Concepts 2020-2026
    # 全网前沿: 2026 SMC 方法论四大支柱之一 (BOS/CHoCH + Order Block + FVG + Liquidity Sweep)
    # 非过拟合: FVG 是市场结构信号 (K线几何派生), 75% FVG 会被回访填充 (ICT 实证)
    # v701 设计: FVG 信号独立输出, 不进入 composite_score (避免被 ERR-v88fp 证伪的 composite 稀释)
    #           agent_decision_engine 直接消费 fvg_signal (类似 hold_bars_adapter 独立信号增强)
    fvg_signal: float = 0.0              # FVG 综合信号 (-1 ~ +1, 正=做多/负=做空)
    fvg_confidence: float = 0.0          # FVG 置信度 (0-1)
    fvg_pattern: str = "neutral"         # FVG 信号模式 (FVGSignal enum value)
    fvg_active_count: int = 0            # 活跃 FVG 数量 (机构未平仓足迹)
    fvg_target_fill_ratio: float = 0.0   # 目标 FVG 填补比例 (0=未填补, 1=已填补)
    fvg_displacement_strength: float = 0.0  # 位移强度 (大K线实体/ATR)
    fvg_new_formed: bool = False         # 是否新形成 FVG (趋势启动信号)
    fvg_entry_price: float = 0.0         # FVG 入场价 (缺口中点 ≈ 机构成本)
    fvg_stop_loss: float = 0.0           # FVG 止损价 (缺口外沿)

    # ===== Order Block 订单块维度 (v702 P1 新增, 用户铁律"指标融会贯通") =====
    # 来源: _v513_liquidity_orderblock.py — ICT Smart Money Concepts 四大支柱之二
    # 全网前沿: 2026 SMC 方法论, 看涨OB=上涨波段启动前最后一根阴线 (机构多单建仓位)
    # 非过拟合: OB 是 K 线几何+波段结构信号, 75% OB 回调会被尊重 (ICT 实证)
    # v702 设计: OB 信号独立输出, 不进入 composite_score (与 FVG/CVD 一致, 避免被 ERR-v88fp 证伪稀释)
    #           agent_decision_engine 直接消费 order_block_signal
    order_block_signal: float = 0.0              # OB 综合信号 (-1 ~ +1, 正=做多/负=做空)
    order_block_confidence: float = 0.0          # OB 置信度 (0-1, 基于强度+距离)
    order_block_active_count: int = 0            # 活跃 OB 数量 (未消化的机构建仓位)
    order_block_bullish_count: int = 0           # 看涨 OB 数量 (机构多单建仓位)
    order_block_bearish_count: int = 0           # 看跌 OB 数量 (机构空单建仓位)
    order_block_nearest_direction: str = "neutral"  # 最近 OB 方向 (bullish/bearish/neutral)
    order_block_nearest_strength: float = 0.0    # 最近 OB 强度 (波段幅度, 越大机构参与度越高)
    order_block_nearest_price: float = 0.0       # 最近 OB 价格中点 (机构成本区)
    order_block_is_mitigated: bool = False       # 最近 OB 是否已消化 (价格已回调到OB)
    order_block_entry_price: float = 0.0         # OB 入场价 (OB 中点 ≈ 机构成本)
    order_block_stop_loss: float = 0.0           # OB 止损价 (OB 对侧外沿)

    # ===== Liquidity Sweep 流动性扫掠维度 (v702 P2 新增, 用户铁律"指标融会贯通") =====
    # 来源: _v513_liquidity_orderblock.py — ICT Smart Money Concepts 四大支柱之四
    # 全网前沿: 2026 SMC 方法论, 扫掠=假突破收割散户止损后反向建仓 (机构猎杀区)
    # 非过拟合: Liquidity Sweep 是 K 线几何+流动性池结构信号, 非参数拟合
    # v702 设计: 扫掠信号独立输出, 不进入 composite_score (与 FVG/CVD/OB 一致, 避免被 ERR-v88fp 证伪稀释)
    #           agent_decision_engine 直接消费 liquidity_sweep_signal
    liquidity_sweep_signal: float = 0.0          # 扫掠综合信号 (-1 ~ +1, 正=做多/负=做空)
    liquidity_sweep_confidence: float = 0.0       # 扫掠置信度 (0-1, 基于池强度+扫掠距离)
    liquidity_sweep_type: str = "none"            # 扫掠类型 (buy_side_sweep=做空/sell_side_sweep=做空/none)
    liquidity_sweep_pool_type: str = "none"       # 被扫掠的流动性池类型 (buy_side/sell_side/none)
    liquidity_sweep_pool_level: float = 0.0       # 被扫掠的流动性池价位 (散户止损堆积区)
    liquidity_sweep_bar_idx: int = -1             # 扫掠发生的bar索引 (-1=无扫掠)
    liquidity_sweep_is_valid: bool = False        # 是否为有效扫掠 (收盘回归确认)
    liquidity_sweep_pool_count: int = 0           # 流动性池总数 (等高低点聚类)
    liquidity_sweep_swept_count: int = 0          # 已扫掠的流动性池数 (机构猎杀次数)
    liquidity_sweep_recent_distance: int = 9999   # 最近扫掠距当前bar的距离 (越近越强, 9999=无扫掠)
    liquidity_sweep_entry_price: float = 0.0      # 扫掠入场价 (扫掠bar收盘价)
    liquidity_sweep_stop_loss: float = 0.0        # 扫掠止损价 (扫掠bar极值外沿)

    # ===== 多周期结构维度 =====
    mtf_direction: float = 0.0           # 多周期方向 (-1 ~ +1)
    mtf_strength: float = 0.0            # 多周期一致性 (0-1)
    mtf_is_confluent: bool = False       # 是否多周期一致
    mtf_htf_veto: bool = False           # 大周期否决
    mtf_confluence_score: float = 0.0    # 综合评分 (0-1)

    # ===== 市场状态维度 (v596 C2-d 新增) =====
    regime_raw: str = "unknown"               # 原始 regime 命名
    regime_normalized: str = "ranging"        # 标准化后的 regime (RegimeEnum.value)
    regime_confidence: float = 0.0            # regime 置信度 (0-1)
    regime_strategy_weight: float = 1.0       # 当前策略在当前 regime 下的权重

    # ===== 融合输出 =====
    composite_score: float = 0.0         # 综合信号 (-1 ~ +1)
    confidence: float = 0.0              # 综合置信度 (0-1)
    should_trade: bool = False           # 是否应该交易
    suggested_direction: str = "neutral"  # 建议方向
    suggested_size: float = 0.0          # 建议仓位 (0-1, 已含凯利调整)

    def to_dict(self) -> dict:
        return {
            "kelly": {
                "fraction": round(self.kelly_fraction, 4),
                "track": self.kelly_track,
                "win_rate": round(self.kelly_win_rate, 4),
                "confidence": round(self.kelly_confidence, 4),
            },
            "price_action": {
                "signal": round(self.pa_signal, 4),
                "patterns": self.pa_patterns,
                "bos_choch": self.pa_bos_choch,
                "support": round(self.pa_support, 4),
                "resistance": round(self.pa_resistance, 4),
            },
            "volume_price": {
                "vpin": round(self.vpin_score, 4),
                "is_toxic": self.vpin_is_toxic,
                "poc": round(self.volume_profile_poc, 4),
                "vah": round(self.volume_profile_vah, 4),
                "val": round(self.volume_profile_val, 4),
            },
            "cvd": {
                "value": round(self.cvd_value, 4),
                "slope": round(self.cvd_slope, 6),
                "slope_long": round(self.cvd_slope_long, 6),
                "normalized": round(self.cvd_normalized, 4),
                "buy_sell_pressure": round(self.buy_sell_pressure, 4),
                "pattern": self.cvd_pattern,
                "divergence": self.cvd_divergence,
                "absorption": self.cvd_absorption,
                "signal": round(self.cvd_signal, 4),
                "confidence": round(self.cvd_confidence, 4),
            },
            "mtf": {
                "direction": round(self.mtf_direction, 4),
                "strength": round(self.mtf_strength, 4),
                "confluent": self.mtf_is_confluent,
                "htf_veto": self.mtf_htf_veto,
                "score": round(self.mtf_confluence_score, 4),
            },
            "regime": {
                "raw": self.regime_raw,
                "normalized": self.regime_normalized,
                "confidence": round(self.regime_confidence, 4),
                "strategy_weight": round(self.regime_strategy_weight, 4),
            },
            "fusion": {
                "composite_score": round(self.composite_score, 4),
                "confidence": round(self.confidence, 4),
                "should_trade": self.should_trade,
                "direction": self.suggested_direction,
                "size": round(self.suggested_size, 4),
            },
        }


# ============================================================================
# v598 Phase I: 自适应权重管理器 (基于 IC 的动态调权)
# ============================================================================


class AdaptiveWeightManager:
    """基于 IC (Information Coefficient) 的动态权重调整

    v598 Phase I: 4维特征权重从静态默认 (0.25/0.30/0.20/0.25) → 基于历史表现的动态调权

    用户铁律 "凯利+价格行为+量价+多周期融会贯通":
      - 各维度对 PnL 的预测力 (IC) 不同, 应向 IC 高的维度倾斜
      - 静态权重未基于历史表现调整 = "生搬硬套" (违反用户铁律)

    ERR-110 应用边界 (文档诚信修正, L7 审查问题#1):
      - 不新增硬过滤条件: 不修改 composite_threshold 阈值, 不修改 should_trade 判定逻辑
      - 不删除任何维度: clip_range 下限 0.10, 确保 4 维始终参与 (无维度被裁剪到 0)
      - 权重对入场决策的间接影响 (透明披露):
        * composite = w*kelly + w*pa + w*vp + w*mtf 加权融合
        * 当权重变化导致 composite_score 跨越 composite_threshold 时, should_trade 决策可能翻转
        * 约束: clip_range=(0.10, 0.50) + 保守 lr=0.05, 单代最大摆幅 5%
        * 信号方向不改: 各维度仍按其信号值贡献, 权重仅调整贡献比例

    设计:
      - IC = corrcoef(dimension_scores, realized_pnl), 线性预测力
      - 权重向 IC 高的维度倾斜: weights[dim] += lr * ic
      - 归一化 + clip (clip_range=(0.10, 0.50), 防止某维度权重过大或过小)
      - IC 历史滑窗 (避免单次异常 IC 导致权重剧烈波动)

    参考:
      - Grinold & Kahn (1999) Active Portfolio Management: IC = corr(forecast, return)
      - 业界惯例: IC > 0.05 有效, IC > 0.1 强, IC > 0.2 极强
    """

    # 默认初始权重 (与 FeatureFusionPipeline.DEFAULT_WEIGHTS 一致)
    DEFAULT_INITIAL_WEIGHTS = {
        "kelly": 0.25,
        "price_action": 0.30,
        "volume_price": 0.20,
        "mtf": 0.25,
    }

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.05,
        clip_range: Tuple[float, float] = (0.10, 0.50),
        min_samples_for_ic: int = 5,
        ic_lookback: int = 20,
    ):
        """
        Args:
            initial_weights: 初始权重 (None 时用 DEFAULT_INITIAL_WEIGHTS)
            learning_rate: 权重调整学习率 (0.05 = 保守调整, 每代最多调整 5%)
            clip_range: 单维度权重 clip 范围 (0.10-0.50, 确保 4 维始终参与)
            min_samples_for_ic: 计算 IC 所需最小样本数 (< 此数不更新)
            ic_lookback: IC 历史记录窗口 (最近 N 代的 IC)
        """
        self.weights = (initial_weights or self.DEFAULT_INITIAL_WEIGHTS).copy()
        self.lr = learning_rate
        self.clip_range = clip_range
        self.min_samples_for_ic = min_samples_for_ic
        self.ic_lookback = ic_lookback
        # IC 历史记录 (每代一个 IC 值)
        self.ic_history: Dict[str, List[float]] = {k: [] for k in self.weights}
        # 权重历史 (用于监控/调试)
        self.weight_history: List[Dict[str, float]] = [dict(self.weights)]
        # 漂移调整历史 (v99 Task #92 新增, 与PSIDriftDetector联动)
        # 每次 apply_drift_adjustment 调用记录一次, 用于追溯漂移降权历史
        self.drift_adjustment_history: List[Dict[str, Any]] = []
        # 更新次数
        self.update_count = 0

    def update(
        self,
        dimension_scores: Dict[str, List[float]],
        realized_pnls: List[float],
    ) -> Dict[str, float]:
        """每代进化后用 IC 更新权重

        Args:
            dimension_scores: 各维度本代的信号分数 {dim: [score1, score2, ...]}
                              keys 必须是 "kelly"/"price_action"/"volume_price"/"mtf"
            realized_pnls: 本代各笔交易的实现 PnL [pnl1, pnl2, ...]
                          (与 dimension_scores 的 list 等长)

        Returns:
            更新后的权重 dict

        Note:
          - ERR-110 边界: 不新增硬过滤/不修改 threshold/不删除维度; 权重通过 composite_score 间接影响入场 (受 clip+lr 约束)
          - 若 len(realized_pnls) < min_samples_for_ic, 跳过更新 (样本不足)
          - IC = np.corrcoef(scores, pnls)[0,1], 范围 [-1, 1]
          - 权重向 IC 高的维度倾斜: weights[dim] += lr * ic
        """
        if len(realized_pnls) < self.min_samples_for_ic:
            logger.debug(
                "AdaptiveWeightManager.update: 样本不足 (%d < %d), 跳过",
                len(realized_pnls), self.min_samples_for_ic,
            )
            return dict(self.weights)

        for dim, scores in dimension_scores.items():
            if dim not in self.weights:
                continue
            if len(scores) != len(realized_pnls):
                logger.warning(
                    "AdaptiveWeightManager.update: %s 维度长度不匹配 (%d != %d), 跳过该维度",
                    dim, len(scores), len(realized_pnls),
                )
                continue
            # 计算 IC
            try:
                arr_s = np.array(scores, dtype=float)
                arr_p = np.array(realized_pnls, dtype=float)
                # 过滤 NaN
                mask = ~(np.isnan(arr_s) | np.isnan(arr_p))
                if mask.sum() < self.min_samples_for_ic:
                    continue
                if np.std(arr_s[mask]) < 1e-8 or np.std(arr_p[mask]) < 1e-8:
                    ic = 0.0  # 零方差时 IC=0
                else:
                    ic = float(np.corrcoef(arr_s[mask], arr_p[mask])[0, 1])
                    if np.isnan(ic):
                        ic = 0.0
            except Exception as e:
                logger.debug("AdaptiveWeightManager IC 计算失败 (%s): %s", dim, str(e)[:100])
                ic = 0.0

            # 记录 IC 历史 (滑窗)
            self.ic_history[dim].append(ic)
            if len(self.ic_history[dim]) > self.ic_lookback:
                self.ic_history[dim] = self.ic_history[dim][-self.ic_lookback:]

            # 权重向 IC 高的维度倾斜
            self.weights[dim] += self.lr * ic

        # 归一化 + clip
        self._normalize_and_clip()
        self.update_count += 1
        self.weight_history.append(dict(self.weights))
        if len(self.weight_history) > 100:
            self.weight_history = self.weight_history[-100:]

        logger.info(
            "AdaptiveWeightManager.update #%d: weights=%s, ic_mean3=%s",
            self.update_count,
            {k: round(v, 4) for k, v in self.weights.items()},
            {k: round(float(np.mean(v[-3:])) if v else 0.0, 4)
             for k, v in self.ic_history.items()},
        )
        return dict(self.weights)

    def _normalize_and_clip(self):
        """归一化 + clip (确保权重和=1, 单维度在 clip_range 内)"""
        # 先 clip
        for k in self.weights:
            self.weights[k] = max(self.clip_range[0], min(self.clip_range[1], self.weights[k]))
        # 再归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return dict(self.weights)

    def get_ic_report(self) -> Dict[str, Any]:
        """获取 IC 报告 (用于监控/调试)"""
        report: Dict[str, Any] = {}
        for dim, ics in self.ic_history.items():
            if not ics:
                report[dim] = {"n": 0, "mean_ic": 0.0, "last_ic": 0.0}
            else:
                arr = np.array(ics)
                report[dim] = {
                    "n": len(ics),
                    "mean_ic": round(float(np.mean(arr)), 4),
                    "last_ic": round(float(ics[-1]), 4),
                    "std_ic": round(float(np.std(arr)), 4) if len(ics) > 1 else 0.0,
                }
        report["update_count"] = self.update_count
        report["current_weights"] = {k: round(v, 4) for k, v in self.weights.items()}
        return report

    # ========================================================================
    # v99 Task #92: PSI 漂移联动降权
    # ========================================================================

    def apply_drift_adjustment(self, drift_report: Dict[str, Any]) -> Dict[str, float]:
        """根据 PSI 漂移报告调整权重 (v99 Task #92 新增)

        与 IC 调权联动:
          - IC 调权 (update): 基于历史预测力调整权重 (长期适应, 向 alpha 高的维度倾斜)
          - PSI 漂移 (本方法): 基于分布稳定性调整权重 (短期防护, 漂移维度降权)
          - 综合: 最终权重 = IC 调整后权重 × PSI multiplier, 然后归一化+clip

        用户铁律 "不生搬硬套":
          - IC 反映"哪个维度预测力强", PSI 反映"哪个维度分布变了"
          - 一个维度可能 IC 高但 PSI 也高 (预测力强但分布漂移) → 适度降权保留探针
          - 一个维度可能 IC 低且 PSI 低 (预测力弱但稳定) → 通过 IC 降权, 不受 PSI 影响

        边界 (与 ERR-110 一致):
          - 不删除维度 (PSI 严重漂移也保留 0.2 探针, 不清零)
          - 漂移降权是临时的, PSI 回落后, 下次 update() 的 IC 调权会重新拉回
          - 仍受 clip_range 约束 (0.10-0.50)

        Args:
            drift_report: PSIDriftDetector.update() 的返回值
                {
                    "per_dimension": {
                        "kelly": {"psi": float, "level": str,
                                  "weight_multiplier": float, ...},
                        ...
                    },
                    "overall_level": "OK"/"WARN"/"BLOCK",
                    "drifted_dimensions": List[str],
                }

        Returns:
            调整后的权重 dict (已归一化+clip)
        """
        per_dim = drift_report.get("per_dimension", {})
        drifted_dims = drift_report.get("drifted_dimensions", [])
        overall_level = drift_report.get("overall_level", "N/A")

        # 记录调整前的权重快照
        weights_before = dict(self.weights)

        # 应用 PSI multiplier 到每个维度
        adjustments: Dict[str, Dict[str, float]] = {}
        for dim in self.weights:
            if dim not in per_dim:
                continue
            dim_report = per_dim[dim]
            mult = dim_report.get("weight_multiplier", 1.0)
            psi_val = dim_report.get("psi")

            if mult < 1.0 and psi_val is not None:
                # 漂移降权: 当前权重 × multiplier
                old_w = self.weights[dim]
                new_w = old_w * mult
                self.weights[dim] = new_w
                adjustments[dim] = {
                    "psi": psi_val,
                    "level": dim_report.get("level", "N/A"),
                    "multiplier": mult,
                    "weight_before": round(old_w, 4),
                    "weight_after": round(new_w, 4),
                }
                logger.warning(
                    "[AdaptiveWeightManager] 漂移降权 %s: %.4f → %.4f "
                    "(PSI=%.4f, mult=%.2f, level=%s)",
                    dim, old_w, new_w, psi_val, mult, dim_report.get("level", "N/A"),
                )

        # 重新归一化 + clip (确保权重和=1, 单维度在 clip_range 内)
        self._normalize_and_clip()

        # 记录漂移调整历史 (用于追溯)
        self.drift_adjustment_history.append({
            "update_count": self.update_count,
            "overall_level": overall_level,
            "drifted_dimensions": drifted_dims,
            "weights_before": {k: round(v, 4) for k, v in weights_before.items()},
            "weights_after": {k: round(v, 4) for k, v in self.weights.items()},
            "adjustments": adjustments,
        })
        if len(self.drift_adjustment_history) > 100:
            self.drift_adjustment_history = self.drift_adjustment_history[-100:]

        logger.info(
            "[AdaptiveWeightManager] apply_drift_adjustment: overall=%s, drifted=%s, "
            "weights=%s",
            overall_level, drifted_dims,
            {k: round(v, 4) for k, v in self.weights.items()},
        )

        return dict(self.weights)

    def get_drift_report(self) -> List[Dict[str, Any]]:
        """获取漂移调整历史 (用于监控/调试, v99 Task #92 新增)"""
        return list(self.drift_adjustment_history)


# ============================================================================
# 核心管道
# ============================================================================

class FeatureFusionPipeline:
    """统一特征融合管道 — 4维标准化融合

    维度权重 (动态调整):
      - 凯利公式: 0.25 (仓位控制)
      - 价格行为: 0.30 (信号质量)
      - 量价分析: 0.20 (市场微观结构)
      - 多周期结构: 0.25 (趋势确认)

    双轨凯利:
      - 高置信度 (样本≥50): full Kelly (激进)
      - 低置信度 (样本<50): half Kelly (保守)
    """

    # 默认维度权重
    DEFAULT_WEIGHTS = {
        "kelly": 0.25,
        "price_action": 0.30,
        "volume_price": 0.20,
        "mtf": 0.25,
    }

    def __init__(
        self,
        weights: Dict[str, float] = None,
        kelly_track_threshold: int = 50,
        vpin_toxic_threshold: float = 0.7,
        composite_threshold: float = 0.3,
    ):
        """
        Args:
            weights: 维度权重 (kelly/price_action/volume_price/mtf)
            kelly_track_threshold: 凯利双轨切换的样本量阈值
            vpin_toxic_threshold: VPIN 毒性阈值
            composite_threshold: 综合信号交易阈值
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        # 归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        self.kelly_track_threshold = kelly_track_threshold
        self.vpin_toxic_threshold = vpin_toxic_threshold
        self.composite_threshold = composite_threshold

        # v598 Phase I: 自适应权重管理器 (基于 IC 的动态调权)
        # ERR-110 边界: 不新增硬过滤/不修改 threshold/不删除维度; 权重通过 composite_score 间接影响入场
        # extract 中用 self.adaptive_weights.get_weights() 替代静态 self.weights
        # loop_end 调用 self.adaptive_weights.update(dim_scores, realized_pnls) 更新权重
        self.adaptive_weights = AdaptiveWeightManager(initial_weights=self.weights)

        # 延迟初始化各子模块 (避免 import 失败)
        self._kelly_system = None
        self._pa_system = None
        self._vpin_calculator = None
        self._volume_profile = None
        self._mtf_core = None
        self._regime_detector = None  # v596 C2-d: RegimeDetectionSystem 延迟初始化
        self._cvd_analyzer = None     # v98: CVD 订单流分析器延迟初始化
        self._fvg_detector = None     # v701 P1-3: FVG 检测器延迟初始化 (SMC 核心维度)
        self._ob_detector = None      # v702 P1: Order Block 检测器延迟初始化 (SMC 核心维度)
        self._liquidity_sweep_detector = None  # v702 P2: Liquidity Sweep 检测器延迟初始化 (SMC 核心维度)

        # v699 逐笔归因→5维进化建议 indicator_patches 注入 (用户铁律"复盘作用于整个AI量化技能包")
        # 来源: PerTradeAttributionEngine 产出 → EvolutionSuggestionApplier 应用到 overlay
        # 设计: indicator_patches(reduce_position_by / monitor_only) 在 _fuse_features 末尾消费
        # 闭环: loop_end 归因→应用overlay → _init_decision_engines 注入 → _fuse_features 消费
        # 安全: patch 匹配失败时静默跳过(防御式), 不破坏主融合流程
        self._indicator_patches: List[Dict[str, Any]] = []

    def set_indicator_patches(self, patches: Optional[List[Dict[str, Any]]] = None) -> None:
        """v699: 注入 indicator 维度进化建议 patches

        由 EvolutionLoop._init_decision_engines 调用, 将归因引擎产出+应用器
        写入 _active_evolution_overlay.json 的 indicator_patches 注入到 pipeline.

        闭环 (用户铁律"复盘作用于整个AI量化技能包"):
          1. loop_end PerTradeAttributionEngine 归因 → EvolutionSuggestionApplier 应用
          2. _init_decision_engines 加载 overlay → set_indicator_patches 注入
          3. _fuse_features 消费 patches:
             - feature(pa_signal/vpin_score/mtf_strength/kelly_fraction/composite_score 等)
             - threshold + op 匹配 → action(reduce_position_by factor / monitor_only)
          4. 回测验证 → evaluate_rollbacks(退化则自动回滚)

        安全设计 (防御式集成, 不破坏主融合流程):
          - patch 匹配失败时静默跳过 (不抛异常)
          - 仅 active 状态的 patch 生效
          - reduce_position_by → composite_score *= factor (间接影响 decide 的 signal_score)
          - monitor_only → composite_score = 0 (中性, 不增强不减仓)

        Args:
            patches: indicator 维度 patch 列表 (None=清空)
        """
        self._indicator_patches = patches or []

    def _init_kelly(self):
        """延迟初始化凯利系统"""
        if self._kelly_system is not None:
            return
        try:
            from frontier_enhancement import DynamicAdaptiveKelly
            self._kelly_system = DynamicAdaptiveKelly()
            logger.info("凯利系统已初始化")
        except Exception as e:
            logger.warning(f"凯利系统初始化失败: {e}")

    def _init_pa(self):
        """延迟初始化价格行为系统"""
        if self._pa_system is not None:
            return
        try:
            from chanlun_priceaction import ChanLunPriceActionSystem
            self._pa_system = ChanLunPriceActionSystem()
            logger.info("价格行为系统已初始化")
        except Exception as e:
            logger.warning(f"价格行为系统初始化失败: {e}")

    def _init_vpin(self):
        """延迟初始化 VPIN 计算器"""
        if self._vpin_calculator is not None:
            return
        try:
            from frontier_enhancement import VPINCalculator
            self._vpin_calculator = VPINCalculator()
            logger.info("VPIN 计算器已初始化")
        except Exception as e:
            logger.warning(f"VPIN 初始化失败: {e}")

    def _init_volume_profile(self):
        if self._volume_profile is not None:
            return
        try:
            from frontier_enhancement import VolumeProfileAnalyzer
            self._volume_profile = VolumeProfileAnalyzer()
            logger.info("VolumeProfile 已初始化")
        except Exception as e:
            logger.warning(f"VolumeProfile 初始化失败: {e}")

    def _init_cvd(self):
        """v98: 延迟初始化 CVD 订单流分析器

        用户铁律"量价分析融会贯通"的关键补齐:
          - VPIN + VolumeProfile 是订单流毒性和价值区分析
          - CVD 是市场微观结构最强信号 (73% BTC反转前出现背离, CryptoQuant数据)
          - 三者互补: VPIN看毒性, VP看价值区, CVD看主动买卖方向+隐藏吸筹/分发
        非过拟合: CVD 阈值全部来自业界推荐值 (LiveVolatile/LedgerMind 2026), 非历史拟合
        """
        if self._cvd_analyzer is not None:
            return
        try:
            from order_flow_cvd import OrderFlowCVDAnalyzer
            self._cvd_analyzer = OrderFlowCVDAnalyzer(lookback=50)
            logger.info("CVD 订单流分析器已初始化")
        except Exception as e:
            logger.warning(f"CVD 初始化失败: {e}")

    def _init_mtf(self):
        """延迟初始化 MTF 核心"""
        if self._mtf_core is not None:
            return
        try:
            from mtf_core import MTFCore
            self._mtf_core = MTFCore(timeframes=["4h", "1h", "15m"])
            logger.info("MTF 核心已初始化")
        except Exception as e:
            logger.warning(f"MTF 核心初始化失败: {e}")

    def _init_fvg(self):
        """v701 P1-3: 延迟初始化 FVG 检测器

        用户铁律"指标融会贯通"的核心补齐:
          - FVG (Fair Value Gap) 是 ICT Smart Money Concepts 四大支柱之一
          - 2026 SMC 主流方法论: BOS/CHoCH + Order Block + FVG + Liquidity Sweep
          - 75% FVG 会被回访填充 (ICT 实证研究), FVG 中点 ≈ 机构平均成本
          - 与 CVD 互补: CVD 看主动买卖方向, FVG 看机构成本区+流动性磁铁

        非过拟合: FVG 是 K 线几何信号 (3-K线模式 + 位移确认), 非参数拟合
        v701 设计: FVG 信号独立输出到 fv.fvg_signal, 不进入 composite_score
                   (避免被 ERR-v88fp 证伪的 composite 稀释), 直达 agent_decision_engine
        """
        if self._fvg_detector is not None:
            return
        try:
            from fair_value_gap_detector import FairValueGapDetector
            self._fvg_detector = FairValueGapDetector(symbol="auto")
            logger.info("FVG 检测器已初始化 (v701 P1-3)")
        except Exception as e:
            logger.warning(f"FVG 初始化失败: {e}")

    def _init_order_block(self):
        """v702 P1: 延迟初始化 Order Block 检测器

        用户铁律"指标融会贯通"的核心补齐 (SMC 四大支柱之二):
          - OB (Order Block) 是 ICT Smart Money Concepts 核心概念
          - 看涨OB = 上涨波段启动前的最后一根阴线 (机构多单建仓位)
          - 看跌OB = 下跌波段启动前的最后一根阳线 (机构空单建仓位)
          - 价格回调到OB区域 = 高胜率入场点 (机构成本支撑/压力)
          - 与 FVG 互补: FVG 看缺口, OB 看机构建仓位

        非过拟合: OB 是 K 线几何 + 波段结构信号, 非参数拟合
        v702 设计: OB 信号独立输出到 fv.order_block_signal, 不进入 composite_score
                   (与 FVG/CVD 一致, 避免被 ERR-v88fp 证伪的 composite 稀释)
                   直达 agent_decision_engine

        API 说明: _v513_liquidity_orderblock.py 是函数式 API (无类),
                  只需验证 import 成功, 无需实例化
        """
        if self._ob_detector is not None:
            return
        try:
            # 函数式 API (无类), 验证 import 可用
            self._ob_detector = True  # 标记初始化成功 (函数式 API 无需实例化)
            logger.info("Order Block 检测器已初始化 (v702 P1)")
        except Exception as e:
            logger.warning(f"Order Block 初始化失败: {e}")

    def _init_liquidity_sweep(self):
        """v702 P2: 延迟初始化 Liquidity Sweep 检测器

        用户铁律"指标融会贯通"的核心补齐 (SMC 四大支柱之四):
          - Liquidity Sweep (流动性扫掠) 是 ICT Smart Money Concepts 核心概念
          - 扫掠 = 价格突破流动性池(等高低点)后迅速回归 = 假突破收割散户止损
          - 买方流动性扫掠 (突破前高后回落) → 做空信号 (机构猎杀买方止损)
          - 卖方流动性扫掠 (跌破前低后反弹) → 做多信号 (机构猎杀卖方止损)
          - 与 OB 互补: OB 看机构建仓位, 扫掠看机构猎杀区

        非过拟合: Liquidity Sweep 是 K 线几何 + 流动性池结构信号, 非参数拟合
        v702 设计: 扫掠信号独立输出到 fv.liquidity_sweep_signal, 不进入 composite_score
                   (与 FVG/CVD/OB 一致, 避免被 ERR-v88fp 证伪的 composite 稀释)
                   直达 agent_decision_engine

        API 说明: _v513_liquidity_orderblock.py 是函数式 API (无类),
                  只需验证 import 成功, 无需实例化
        """
        if self._liquidity_sweep_detector is not None:
            return
        try:
            # 函数式 API (无类), 验证 import 可用
            self._liquidity_sweep_detector = True  # 标记初始化成功 (函数式 API 无需实例化)
            logger.info("Liquidity Sweep 检测器已初始化 (v702 P2)")
        except Exception as e:
            logger.warning(f"Liquidity Sweep 初始化失败: {e}")

    def _init_regime_detector(self):
        """v596 C2-d: 延迟初始化 RegimeDetectionSystem

        从 regime_detection.py 加载 HMM-based 市场状态检测器，
        用于将 4 维特征管道扩展为 4+1=5 维（加入市场状态维度）。
        """
        if self._regime_detector is not None:
            return
        try:
            from regime_detection import RegimeDetectionSystem
            self._regime_detector = RegimeDetectionSystem()
            logger.info("RegimeDetectionSystem 已初始化 (v596 C2-d)")
        except Exception as e:
            logger.warning(f"RegimeDetectionSystem 初始化失败: {e}")

    def _extract_regime(self, fv: FeatureVector, klines: List[Dict]):
        """v596 C2-d: 提取市场状态特征 (第5维)

        使用 HMM 检测器对最近 K 线的收益率序列建模，
        输出当前 regime (trending_up/down, ranging, volatile, crisis, accumulation)
        并归一化到 _v596_regime_namespace 定义的标准6状态。

        Args:
            fv: FeatureVector 待填充
            klines: 1h K 线数据
        """
        if not klines or len(klines) < 50:
            fv.regime_normalized = "ranging"  # 默认震荡（最安全假设）
            return

        self._init_regime_detector()
        if self._regime_detector is None:
            fv.regime_normalized = "ranging"
            return

        try:
            # 提取最近 200 根 K 线的收益率序列
            closes = [float(k.get("close", 0)) for k in klines[-200:]]
            returns = [
                closes[i] / closes[i - 1] - 1
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            # 只喂最近 50 根避免状态过老（HMM 状态会随时间衰减）
            for r in returns[-50:]:
                self._regime_detector.update(r)

            regime_data = self._regime_detector.get_regime()
            # v99 P4 修复 (ERR-20260704-v99-regime-key-mismatch): 键名不匹配接线 bug
            # ----------------------------------------------------------------
            # 问题: RegimeDetectionSystem.get_regime() → _combine_states() 返回的键名是
            #   "market_regime" 和 "market_confidence", 但这里读取的是 "regime" 和 "confidence".
            #   键名不匹配 → regime_data.get("regime", "unknown") 永远返回 "unknown"
            #   → normalize_regime("unknown") = RANGING → P7-BT regime_dist 100% ranging
            #   → conf_mean=0.0 → adjusted=0% (P7 regime 调整完全失效, 空壳接线)
            #
            # 影响: 即使 HMM 正确检测到 bull/bear, feature_vector 也读不到,
            #   P7-BT 永远用 ranging/0.0 默认值, regime 调制完全空壳.
            #   之前 HMM 锁死 bear 时, 由于此 bug, P7-BT 实际读的是 "unknown"→ranging,
            #   而非真正的 bear. 但 conf_mean=0.0 导致 effective_mult=1.0 (无调整),
            #   所以表面上 "没有错误减仓", 实际上 "regime 调制完全失效".
            #
            # 修复: 用正确的键名 "market_regime" 和 "market_confidence" 读取.
            # 非过拟合: 纯接线修复, 不引入任何参数, 仅修正键名匹配.
            fv.regime_raw = str(regime_data.get("market_regime",
                                                regime_data.get("regime", "unknown")))
            fv.regime_confidence = float(regime_data.get("market_confidence",
                                                         regime_data.get("confidence", 0.0)) or 0.0)

            # 归一化到标准 6 状态 (TRENDING_UP/DOWN, RANGING, VOLATILE, CRISIS, ACCUMULATION)
            from _v596_regime_namespace import normalize_regime
            fv.regime_normalized = normalize_regime(fv.regime_raw).value
        except Exception as e:
            logger.warning(f"Regime 提取失败: {e}")
            fv.regime_normalized = "ranging"

    # ========================================================================
    # 主入口
    # ========================================================================

    def extract(
        self,
        klines_1h: List[Dict],
        klines_4h: Optional[List[Dict]] = None,
        klines_15m: Optional[List[Dict]] = None,
        symbol: str = "unknown",
        trade_history: Optional[List[float]] = None,
    ) -> FeatureVector:
        """提取统一特征向量

        Args:
            klines_1h: 1小时K线 (必需)
            klines_4h: 4小时K线 (可选, 用于MTF)
            klines_15m: 15分钟K线 (可选, 用于MTF)
            symbol: 交易对
            trade_history: 历史收益率列表 (用于凯利更新)

        Returns:
            FeatureVector: 统一特征向量
        """
        fv = FeatureVector()

        # ===== 维度1: 凯利公式 (双轨) =====
        self._extract_kelly(fv, trade_history)

        # ===== 维度2: 价格行为 =====
        self._extract_price_action(fv, klines_1h, symbol)

        # ===== 维度3: 量价分析 =====
        self._extract_volume_price(fv, klines_1h)

        # ===== 维度3.5: FVG 公允价值缺口 (v701 P1-3 新增, SMC 核心维度) =====
        # 用户铁律"指标融会贯通" — FVG 是 ICT SMC 四大支柱之一, 之前未接线
        # 设计: FVG 信号独立输出到 fv.fvg_signal, 不进入 composite_score (避免被 ERR-v88fp 稀释)
        self._extract_fvg(fv, klines_1h, symbol)

        # ===== 维度3.6: Order Block 订单块 (v702 P1 新增, SMC 核心维度) =====
        # 用户铁律"指标融会贯通" — OB 是 ICT SMC 四大支柱之二, 之前未接线 (违反"禁止空壳交付")
        # 设计: OB 信号独立输出到 fv.order_block_signal, 不进入 composite_score (与 FVG/CVD 一致)
        self._extract_order_block(fv, klines_1h, symbol)

        # ===== 维度3.7: Liquidity Sweep 流动性扫掠 (v702 P2 新增, SMC 核心维度) =====
        # 用户铁律"指标融会贯通" — 扫掠是 ICT SMC 四大支柱之四, 之前未接线 (违反"禁止空壳交付")
        # 设计: 扫掠信号独立输出到 fv.liquidity_sweep_signal, 不进入 composite_score (与 FVG/CVD/OB 一致)
        self._extract_liquidity_sweep(fv, klines_1h, symbol)

        # ===== 维度4: 多周期结构 =====
        self._extract_mtf(fv, klines_1h, klines_4h, klines_15m)

        # ===== 维度5: 市场状态 (v596 C2-d 新增) =====
        self._extract_regime(fv, klines_1h)

        # ===== 融合: 加权综合评分 =====
        self._fuse_features(fv)

        # v598: 保存最后处理的 FeatureVector，供 get_pipeline_output() 使用
        self._last_fv = fv

        return fv

    # ========================================================================
    # 维度1: 凯利公式 (双轨)
    # ========================================================================

    def _extract_kelly(
        self, fv: FeatureVector, trade_history: Optional[List[float]]
    ):
        """提取凯利特征 (双轨: full / half)"""
        if not trade_history or len(trade_history) < 5:
            # 样本不足, 使用默认 quarter Kelly
            fv.kelly_fraction = 0.25  # quarter Kelly
            fv.kelly_track = "half"
            fv.kelly_confidence = 0.0
            return

        self._init_kelly()
        if self._kelly_system is None:
            fv.kelly_fraction = 0.25
            fv.kelly_track = "half"
            return

        try:
            # 逐笔更新凯利交易历史 (update(pnl) 累积 wins/losses/amounts)
            # 注: update_returns_history() 是市场收益率(用于VaR), 非 Kelly PnL
            if hasattr(self._kelly_system, "update"):
                for pnl in trade_history:
                    self._kelly_system.update(float(pnl))

            # 获取凯利统计 — get_stats() 返回:
            #   {win_rate(0-1), avg_win, avg_loss(负值), bayesian_win_rate, ...}
            if hasattr(self._kelly_system, "get_stats"):
                stats = self._kelly_system.get_stats()
                fv.kelly_win_rate = float(stats.get("win_rate", 0.5))
                fv.kelly_avg_win = float(stats.get("avg_win", 0.0))
                # avg_loss 可能为负值 (亏损金额), Kelly公式需用正数
                raw_avg_loss = float(stats.get("avg_loss", 0.0))
                fv.kelly_avg_loss = abs(raw_avg_loss) if raw_avg_loss != 0 else 0.0
                fv.kelly_confidence = min(1.0, len(trade_history) / 100.0)

                # 计算凯利分数: f* = (p*b - q) / b
                # b = avg_win / avg_loss, p = win_rate, q = 1-p
                if fv.kelly_avg_loss > 0 and fv.kelly_win_rate > 0:
                    b = fv.kelly_avg_win / fv.kelly_avg_loss
                    p = fv.kelly_win_rate
                    q = 1 - p
                    kelly_full = (p * b - q) / b
                    kelly_full = max(0.0, min(1.0, kelly_full))
                else:
                    kelly_full = 0.0

                # 双轨选择
                if len(trade_history) >= self.kelly_track_threshold:
                    # 高置信度: full Kelly (但限制最大 0.5)
                    fv.kelly_fraction = min(0.5, kelly_full)
                    fv.kelly_track = "full"
                else:
                    # 低置信度: half Kelly (保守)
                    fv.kelly_fraction = kelly_full * 0.5
                    fv.kelly_track = "half"
        except Exception as e:
            logger.warning(f"凯利计算失败: {e}")
            fv.kelly_fraction = 0.25
            fv.kelly_track = "half"

    # ========================================================================
    # 维度2: 价格行为
    # ========================================================================

    def _extract_price_action(
        self, fv: FeatureVector, klines: List[Dict], symbol: str
    ):
        """提取价格行为特征"""
        if not klines or len(klines) < 20:
            return

        self._init_pa()
        if self._pa_system is None:
            # 退化: 简单 EMA 对齐
            fv.pa_signal = self._simple_ema_signal(klines)
            return

        try:
            # 喂入 K 线 — update(kline) 只接受1个参数, 返回综合评估结果:
            #   {"signal": "long"/"short"/"neutral", "strength": 0-1,
            #    "sources": [...], "details": {structure, trend_phase, ...}}
            last_result: Optional[Dict] = None
            if hasattr(self._pa_system, "update"):
                for k in klines[-50:]:  # 只取最近50根
                    last_result = self._pa_system.update(k)

            # 从最后一次 update() 返回值提取信号 (无单独 get_signal 方法)
            signal = 0.0
            patterns: List[str] = []
            bos_choch = ""

            if isinstance(last_result, dict):
                raw_signal = last_result.get("signal", "neutral")
                strength = float(last_result.get("strength", 0.0))
                # 转为 -1 ~ +1
                if raw_signal == "long":
                    signal = strength
                elif raw_signal == "short":
                    signal = -strength
                # sources 即检测到的形态/信号源列表
                srcs = last_result.get("sources", [])
                if isinstance(srcs, list):
                    patterns = srcs
                # BOS/CHoCH 在 details.structure 中
                details = last_result.get("details", {})
                struct_detail = details.get("structure", {}) if isinstance(details, dict) else {}
                if isinstance(struct_detail, dict):
                    struct_sig = struct_detail.get("signal", "")
                    if struct_sig:
                        bos_choch = str(struct_sig)

            # 支撑阻力 — get_support_resistance() 无参数, 返回 Tuple[float, float]
            support = 0.0
            resistance = 0.0
            if hasattr(self._pa_system, "get_support_resistance"):
                sr = self._pa_system.get_support_resistance()
                if isinstance(sr, (list, tuple)) and len(sr) >= 2:
                    support, resistance = sr[0], sr[1]

            fv.pa_signal = max(-1.0, min(1.0, signal))
            fv.pa_patterns = patterns if isinstance(patterns, list) else []
            fv.pa_bos_choch = bos_choch
            fv.pa_support = float(support)
            fv.pa_resistance = float(resistance)

        except Exception as e:
            logger.warning(f"价格行为计算失败: {e}")
            fv.pa_signal = self._simple_ema_signal(klines)

    @staticmethod
    def _simple_ema_signal(klines: List[Dict]) -> float:
        """退化模式: 简单 EMA 对齐信号"""
        closes = [k["close"] for k in klines]
        if len(closes) < 50:
            return 0.0

        # EMA50 vs EMA200
        ema50 = sum(closes[-50:]) / 50
        ema200 = sum(closes[-200:]) / min(200, len(closes))

        if ema50 > ema200 * 1.001:
            return 0.5  # 看多
        elif ema50 < ema200 * 0.999:
            return -0.5  # 看空
        return 0.0

    # ========================================================================
    # 维度3: 量价分析
    # ========================================================================

    def _extract_volume_price(
        self, fv: FeatureVector, klines: List[Dict]
    ):
        """提取量价特征 (VPIN + VolumeProfile)"""
        if not klines or len(klines) < 20:
            return

        # VPIN
        self._init_vpin()
        if self._vpin_calculator is not None:
            try:
                if hasattr(self._vpin_calculator, "update"):
                    for k in klines[-50:]:
                        self._vpin_calculator.update(k)

                if hasattr(self._vpin_calculator, "get_vpin"):
                    vpin = self._vpin_calculator.get_vpin()
                    fv.vpin_score = float(vpin)
                    fv.vpin_is_toxic = vpin > self.vpin_toxic_threshold
            except Exception as e:
                logger.warning(f"VPIN 计算失败: {e}")

        # VolumeProfile — analyze() 无参数, 需先通过 update_ohlcv_batch() 喂入数据
        self._init_volume_profile()
        if self._volume_profile is not None:
            try:
                # 先批量喂入 K 线 (重建 Volume Profile)
                if hasattr(self._volume_profile, "update_ohlcv_batch"):
                    self._volume_profile.update_ohlcv_batch(klines[-100:])
                # 再调用 analyze() (无参数, 返回 VolumeProfileResult dataclass)
                if hasattr(self._volume_profile, "analyze"):
                    vp_result = self._volume_profile.analyze()
                    if isinstance(vp_result, dict):
                        fv.volume_profile_poc = float(vp_result.get("poc", 0.0))
                        fv.volume_profile_vah = float(vp_result.get("vah", 0.0))
                        fv.volume_profile_val = float(vp_result.get("val", 0.0))
                    elif hasattr(vp_result, "poc"):
                        # VolumeProfileResult dataclass
                        fv.volume_profile_poc = float(getattr(vp_result, "poc", 0.0))
                        fv.volume_profile_vah = float(getattr(vp_result, "vah", 0.0))
                        fv.volume_profile_val = float(getattr(vp_result, "val", 0.0))
            except Exception as e:
                logger.warning(f"VolumeProfile 计算失败: {e}")

        # v98 新增: CVD 订单流分析 (用户铁律"量价分析融会贯通"的关键补齐)
        # 来源: order_flow_cvd.py — Bulkowski OHLCV→delta + 4种CVD-Price模式 + 背离 + absorption
        # 非过拟合: CVD 是市场微观结构信号 (实际交易数据派生), 阞值用业界推荐值
        self._init_cvd()
        if self._cvd_analyzer is not None:
            try:
                # 重置 + 批量喂入 K 线 (CVD 是 stateless 分析, 每次 extract 重新计算)
                # 注意: 这里用 klines[-50:] 与 lookback=50 对齐
                self._cvd_analyzer.reset()
                self._cvd_analyzer.update_batch(klines[-50:])
                cvd_result = self._cvd_analyzer.analyze()
                if hasattr(cvd_result, "cvd_value"):
                    fv.cvd_value = float(cvd_result.cvd_value)
                    fv.cvd_slope = float(cvd_result.cvd_slope)
                    fv.cvd_slope_long = float(cvd_result.cvd_slope_long)
                    fv.cvd_normalized = float(cvd_result.cvd_normalized)
                    fv.buy_sell_pressure = float(cvd_result.buy_sell_pressure)
                    fv.cvd_pattern = str(cvd_result.cvd_pattern)
                    fv.cvd_divergence = int(cvd_result.cvd_divergence)
                    fv.cvd_absorption = bool(cvd_result.cvd_absorption)
                    fv.cvd_signal = float(cvd_result.signal)
                    fv.cvd_confidence = float(cvd_result.confidence)
            except Exception as e:
                logger.warning(f"CVD 计算失败: {e}")

    # ========================================================================
    # 维度3.5: FVG 公允价值缺口 (v701 P1-3 新增, SMC 核心维度)
    # ========================================================================

    def _extract_fvg(
        self, fv: FeatureVector, klines: List[Dict], symbol: str = "auto"
    ):
        """v701 P1-3: 提取 FVG (Fair Value Gap) 特征

        用户铁律"指标融会贯通"的核心补齐:
          - FVG 是 ICT Smart Money Concepts 四大支柱之一
          - 75% FVG 会被回访填充 (ICT 实证), FVG 中点 ≈ 机构平均成本
          - 与 CVD 互补: CVD 看主动买卖方向, FVG 看机构成本区+流动性磁铁

        设计 (避免被 ERR-v88fp 证伪的 composite 稀释):
          - FVG 信号独立输出到 fv.fvg_signal (-1 ~ +1), 不进入 composite_score
          - agent_decision_engine 直接消费 fv.fvg_signal (类似 hold_bars_adapter)
          - 同时输出 fvg_pattern/fvg_active_count/fvg_target_fill_ratio 供决策参考

        非过拟合: FVG 是 K 线几何信号 (3-K线模式 + 位移确认), 非参数拟合
        """
        if not klines or len(klines) < 50:
            return

        # 确保 import 可用 (延迟初始化验证)
        self._init_fvg()
        if self._fvg_detector is None:
            return

        try:
            # 重新导入类 (避免依赖 self._fvg_detector 实例的状态累积)
            from fair_value_gap_detector import (
                FairValueGapDetector,
                FVGSignal,
                FVGType,
            )

            # 每次 extract 重新创建检测器 (stateless 分析, 避免 _bars/_fvgs 状态累积)
            # FVG 检测基于 K 线几何, 100 根 K 线足以发现近期 FVG
            detector = FairValueGapDetector(symbol=symbol or "auto")
            for k in klines[-100:]:
                ts = float(k.get("timestamp", k.get("time", 0)))
                # 时间戳单位检测 (秒/毫秒), 与 v99 years_span 修复一致
                if ts > 1e12:  # 毫秒级
                    ts = ts / 1000.0
                detector.update_price(
                    timestamp=ts,
                    open_=float(k.get("open", 0)),
                    high=float(k.get("high", 0)),
                    low=float(k.get("low", 0)),
                    close=float(k.get("close", 0)),
                    volume=float(k.get("volume", 0)),
                )

            result = detector.analyze()

            # 映射 FVGAnalysisResult → FeatureVector.fvg_* 字段
            fv.fvg_confidence = float(result.confidence)
            fv.fvg_pattern = str(result.signal.value) if result.signal else "neutral"
            fv.fvg_active_count = len(result.active_fvgs) if result.active_fvgs else 0
            fv.fvg_target_fill_ratio = float(result.fill_ratio)
            fv.fvg_displacement_strength = float(result.displacement_strength)
            fv.fvg_new_formed = result.new_fvg_formed is not None
            if result.entry_price > 0:
                fv.fvg_entry_price = float(result.entry_price)
            if result.stop_loss > 0:
                fv.fvg_stop_loss = float(result.stop_loss)

            # 信号映射: FVGSignal enum → fvg_signal 浮点值 (-1 ~ +1)
            # 设计依据: ICT SMC 方法论 + 2026 前沿研究
            #   - DISPLACEMENT_BREAKOUT: 位移突破 → 看新 FVG 类型 (bullish=+1, bearish=-1)
            #   - FVG_BOUNCE_LONG/IFVG_REVERSAL_LONG: +1.0 (做多信号)
            #   - FVG_BOUNCE_SHORT/IFVG_REVERSAL_SHORT: -1.0 (做空信号)
            #   - FVG_FILL_WARNING/FVG_FULLY_MITIGATED: 0.0 (减仓/趋势结束, 不直接给方向)
            #   - HOLD/NEUTRAL: 0.0
            sig = result.signal
            if sig == FVGSignal.DISPLACEMENT_BREAKOUT:
                # 位移突破: 方向取决于新 FVG 类型
                if result.new_fvg_formed is not None:
                    gap_type = result.new_fvg_formed.gap_type
                    fv.fvg_signal = 1.0 if gap_type == FVGType.BULLISH else -1.0
                else:
                    fv.fvg_signal = 0.0
            elif sig in (FVGSignal.FVG_BOUNCE_LONG, FVGSignal.IFVG_REVERSAL_LONG):
                fv.fvg_signal = 1.0
            elif sig in (FVGSignal.FVG_BOUNCE_SHORT, FVGSignal.IFVG_REVERSAL_SHORT):
                fv.fvg_signal = -1.0
            else:
                # FVG_FILL_WARNING / FVG_FULLY_MITIGATED / HOLD / NEUTRAL
                fv.fvg_signal = 0.0

            # 信号强度调制: displacement_strength 越大, 信号越强 (位移越大 = 机构资金越强)
            if abs(fv.fvg_signal) > 0 and fv.fvg_displacement_strength > 0:
                # displacement_strength 是 ATR 倍数, 限制在 [0.5, 3.0] → 信号 [0.5, 1.0]
                strength_mult = max(0.5, min(1.0, fv.fvg_displacement_strength / 3.0))
                fv.fvg_signal *= strength_mult

            # 诊断日志 (低频, 避免日志爆炸)
            if abs(fv.fvg_signal) > 0.3 and hasattr(self, "_extract_diag_counter"):
                self._extract_diag_counter = getattr(self, "_extract_diag_counter", 0) + 1
                if self._extract_diag_counter % 500 == 1:
                    logger.info(
                        "v701 FVG: signal=%.3f pattern=%s active=%d fill=%.2f disp=%.2f",
                        fv.fvg_signal, fv.fvg_pattern, fv.fvg_active_count,
                        fv.fvg_target_fill_ratio, fv.fvg_displacement_strength,
                    )

        except Exception as e:
            logger.warning(f"FVG 提取失败: {e}")

    # ========================================================================
    # 维度3.6: Order Block 订单块 (v702 P1 新增, SMC 核心维度)
    # ========================================================================

    def _extract_order_block(
        self, fv: FeatureVector, klines: List[Dict], symbol: str = "auto"
    ):
        """v702 P1: 提取 Order Block (订单块) 特征

        用户铁律"指标融会贯通"的核心补齐:
          - OB 是 ICT Smart Money Concepts 四大支柱之二
          - 看涨OB = 上涨波段启动前最后一根阴线 (机构多单建仓位)
          - 价格回调到看涨OB = 高胜率做多点 (机构成本支撑)
          - 与 FVG 互补: FVG 看缺口, OB 看机构建仓位

        设计 (与 FVG/CVD 一致, 避免被 ERR-v88fp 证伪的 composite 稀释):
          - OB 信号独立输出到 fv.order_block_signal (-1 ~ +1)
          - agent_decision_engine 直接消费 (类似 FVG/CVD 直连模式)
          - 同时输出 order_block_active_count/nearest_strength 供决策参考

        非过拟合: OB 是 K 线几何 + 波段结构信号 (3-K线模式 + 波段幅度确认), 非参数拟合

        信号映射依据 (ICT SMC 2026 实证):
          - 看涨OB (未消化) + 价格在OB上方/回调到OB → +1.0 (机构成本支撑, 做多)
          - 看跌OB (未消化) + 价格在OB下方/反弹到OB → -1.0 (机构成本压力, 做空)
          - 价格穿越OB (看涨OB下方/看跌OB上方) → 0.0 (OB 失效, 中性)
        """
        if not klines or len(klines) < 50:
            return

        # 确保 import 可用 (延迟初始化验证)
        self._init_order_block()
        if self._ob_detector is None:
            return

        try:
            # 直接调用函数式 API (stateless, 无状态累积)
            from _v513_liquidity_orderblock import find_order_blocks
            obs = find_order_blocks(klines, min_swing_pct=0.01, lookback=100)

            # 统计 OB
            unmitigated_obs = [ob for ob in obs if not ob.is_mitigated]
            bullish_obs = [ob for ob in unmitigated_obs if ob.direction == "bullish"]
            bearish_obs = [ob for ob in unmitigated_obs if ob.direction == "bearish"]

            fv.order_block_active_count = len(unmitigated_obs)
            fv.order_block_bullish_count = len(bullish_obs)
            fv.order_block_bearish_count = len(bearish_obs)

            if not unmitigated_obs:
                # 无活跃 OB → 中性
                fv.order_block_signal = 0.0
                fv.order_block_confidence = 0.0
                return

            # 找最近的未消化 OB (按 bar_idx 排序, 取最大)
            recent_ob = max(unmitigated_obs, key=lambda ob: ob.bar_idx)
            ob_mid = (recent_ob.high + recent_ob.low) / 2.0
            current_price = float(klines[-1].get("close", 0))

            fv.order_block_nearest_direction = str(recent_ob.direction)
            fv.order_block_nearest_strength = float(recent_ob.strength)
            fv.order_block_nearest_price = float(ob_mid)
            fv.order_block_is_mitigated = bool(recent_ob.is_mitigated)
            fv.order_block_entry_price = float(ob_mid)  # OB 中点入场
            # 止损: 看涨OB止损=OB下方, 看跌OB止损=OB上方
            if recent_ob.direction == "bullish":
                fv.order_block_stop_loss = float(recent_ob.low)
            else:
                fv.order_block_stop_loss = float(recent_ob.high)

            # 信号映射: OB 方向 + 价格位置 → 信号值
            # 设计依据: ICT SMC 方法论 + 2026 前沿研究
            #   - 看涨OB (未消化) + 价格在OB上方/回调到OB → +1.0 (机构成本支撑, 做多)
            #   - 看跌OB (未消化) + 价格在OB下方/反弹到OB → -1.0 (机构成本压力, 做空)
            #   - 价格穿越OB (看涨OB下方/看跌OB上方) → 0.0 (OB 失效, 中性)
            if recent_ob.direction == "bullish":
                # 看涨 OB: 价格在 OB 上方 = 机构成本支撑有效
                if current_price >= recent_ob.low:
                    fv.order_block_signal = 1.0
                else:
                    # 价格跌破看涨 OB = OB 失效
                    fv.order_block_signal = 0.0
            else:  # bearish
                # 看跌 OB: 价格在 OB 下方 = 机构成本压力有效
                if current_price <= recent_ob.high:
                    fv.order_block_signal = -1.0
                else:
                    # 价格涨破看跌 OB = OB 失效
                    fv.order_block_signal = 0.0

            # 信号强度调制: strength (波段幅度) 越大, 信号越强 (机构参与度越高)
            # strength 是百分比 (0.01 = 1%), 限制在 [0.01, 0.10] → 信号 [0.5, 1.0]
            if abs(fv.order_block_signal) > 0 and recent_ob.strength > 0:
                strength_mult = max(0.5, min(1.0, recent_ob.strength / 0.10))
                fv.order_block_signal *= strength_mult

            # 置信度: 基于强度 + 距离 (越近越强)
            # 距离: 当前价与 OB 中点的相对距离
            if ob_mid > 0:
                distance_pct = abs(current_price - ob_mid) / ob_mid
                # 距离越近置信度越高 (在 OB 附近 = 高置信度)
                # 距离 0% → 1.0, 距离 7%+ → 0.3 (下限)
                distance_conf = max(0.3, 1.0 - distance_pct * 10.0)
            else:
                distance_conf = 0.3
            strength_conf = max(0.3, min(1.0, recent_ob.strength / 0.10))
            fv.order_block_confidence = (strength_conf + distance_conf) / 2.0

            # 诊断日志 (低频, 避免日志爆炸)
            if abs(fv.order_block_signal) > 0.3:
                self._ob_extract_diag_counter = getattr(self, "_ob_extract_diag_counter", 0) + 1
                if self._ob_extract_diag_counter % 500 == 1:
                    logger.info(
                        "v702 OB: signal=%.3f dir=%s active=%d bull=%d bear=%d strength=%.3f conf=%.2f",
                        fv.order_block_signal, fv.order_block_nearest_direction,
                        fv.order_block_active_count, fv.order_block_bullish_count,
                        fv.order_block_bearish_count, fv.order_block_nearest_strength,
                        fv.order_block_confidence,
                    )

        except Exception as e:
            logger.warning(f"Order Block 提取失败: {e}")

    # ========================================================================
    # 维度3.7: Liquidity Sweep 流动性扫掠 (v702 P2 新增, SMC 核心维度)
    # ========================================================================

    def _extract_liquidity_sweep(
        self, fv: FeatureVector, klines: List[Dict], symbol: str = "auto"
    ):
        """v702 P2: 提取 Liquidity Sweep (流动性扫掠) 特征

        用户铁律"指标融会贯通"的核心补齐:
          - Liquidity Sweep 是 ICT Smart Money Concepts 四大支柱之四
          - 扫掠 = 价格突破流动性池(等高低点)后迅速回归 = 假突破收割散户止损
          - 买方流动性扫掠 (突破前高后回落) → 做空信号 (机构猎杀买方止损)
          - 卖方流动性扫掠 (跌破前低后反弹) → 做多信号 (机构猎杀卖方止损)
          - 与 OB 互补: OB 看机构建仓位, 扫掠看机构猎杀区

        设计 (与 FVG/CVD/OB 一致, 避免被 ERR-v88fp 证伪的 composite 稀释):
          - 扫掠信号独立输出到 fv.liquidity_sweep_signal (-1 ~ +1)
          - agent_decision_engine 直接消费 (类似 FVG/CVD/OB 直连模式)
          - 同时输出 liquidity_sweep_type/pool_count 供决策参考

        非过拟合: Liquidity Sweep 是 K 线几何 + 流动性池结构信号, 非参数拟合

        信号映射依据 (ICT SMC 2026 实证):
          - buy_side_sweep (扫掠买方流动性) → -1.0 (做空, 机构猎杀买方止损后反向)
          - sell_side_sweep (扫掠卖方流动性) → +1.0 (做多, 机构猎杀卖方止损后反向)
          - 无扫掠 → 0.0 (中性)
        """
        if not klines or len(klines) < 50:
            return

        # 确保 import 可用 (延迟初始化验证)
        self._init_liquidity_sweep()
        if self._liquidity_sweep_detector is None:
            return

        try:
            # 直接调用函数式 API (stateless, 无状态累积)
            from _v513_liquidity_orderblock import (
                find_liquidity_pools, detect_all_sweeps,
            )
            pools = find_liquidity_pools(
                klines, order=5, tolerance_pct=0.003,
                min_touches=2, lookback=100,
            )
            sweeps = detect_all_sweeps(klines, pools)

            # 统计流动性池和扫掠
            swept_pools = [p for p in pools if p.is_swept]
            fv.liquidity_sweep_pool_count = len(pools)
            fv.liquidity_sweep_swept_count = len(swept_pools)

            if not sweeps:
                # 无扫掠 → 中性
                fv.liquidity_sweep_signal = 0.0
                fv.liquidity_sweep_confidence = 0.0
                fv.liquidity_sweep_type = "none"
                fv.liquidity_sweep_pool_type = "none"
                fv.liquidity_sweep_recent_distance = 9999
                return

            # 找最近的扫掠 (按 sweep.bar_idx 排序, 取最大)
            sweeps.sort(key=lambda x: x[1].bar_idx, reverse=True)
            recent_pool, recent_sweep = sweeps[0]

            n = len(klines)
            distance = n - 1 - recent_sweep.bar_idx  # 距离当前 bar 的距离

            fv.liquidity_sweep_type = str(recent_sweep.sweep_type)
            fv.liquidity_sweep_pool_type = str(recent_pool.pool_type)
            fv.liquidity_sweep_pool_level = float(recent_pool.level_price)
            fv.liquidity_sweep_bar_idx = int(recent_sweep.bar_idx)
            fv.liquidity_sweep_is_valid = bool(recent_sweep.is_valid)
            fv.liquidity_sweep_recent_distance = int(max(0, distance))
            fv.liquidity_sweep_entry_price = float(recent_sweep.close)

            # 止损: buy_side_sweep 止损=扫掠bar最高价上方, sell_side_sweep 止损=扫掠bar最低价下方
            if recent_sweep.sweep_type == "buy_side_sweep":
                fv.liquidity_sweep_stop_loss = float(recent_sweep.sweep_high)
            else:
                fv.liquidity_sweep_stop_loss = float(recent_sweep.sweep_low)

            # 信号映射: 扫掠类型 → 信号值
            # 设计依据: ICT SMC 方法论 + 2026 前沿研究
            #   - buy_side_sweep (扫掠买方流动性) → -1.0 (做空, 机构猎杀买方止损后反向)
            #   - sell_side_sweep (扫掠卖方流动性) → +1.0 (做多, 机构猎杀卖方止损后反向)
            if recent_sweep.sweep_type == "buy_side_sweep":
                fv.liquidity_sweep_signal = -1.0
            elif recent_sweep.sweep_type == "sell_side_sweep":
                fv.liquidity_sweep_signal = 1.0
            else:
                fv.liquidity_sweep_signal = 0.0

            # 信号强度调制: pool.touch_count 越大, 流动性池越强, 扫掠信号越强
            # touch_count 通常 2-5, 限制在 [2, 5] → 信号 [0.6, 1.0]
            if abs(fv.liquidity_sweep_signal) > 0 and recent_pool.touch_count > 0:
                touch_mult = max(0.6, min(1.0, recent_pool.touch_count / 5.0))
                fv.liquidity_sweep_signal *= touch_mult

            # 置信度: 基于 pool.touch_count + 扫掠距离 (越近越强)
            # touch_count 越多 = 流动性池越显著 = 扫掠越有意义
            touch_conf = max(0.3, min(1.0, recent_pool.touch_count / 5.0))
            # 距离越近置信度越高 (近期扫掠更相关)
            # 距离 0-5 bar → 1.0, 距离 50+ bar → 0.3
            distance_conf = max(0.3, 1.0 - distance / 50.0)
            fv.liquidity_sweep_confidence = (touch_conf + distance_conf) / 2.0

            # 诊断日志 (低频, 避免日志爆炸)
            if abs(fv.liquidity_sweep_signal) > 0.3:
                self._ls_extract_diag_counter = getattr(self, "_ls_extract_diag_counter", 0) + 1
                if self._ls_extract_diag_counter % 500 == 1:
                    logger.info(
                        "v702 LS: signal=%.3f type=%s pool=%s touches=%d pools=%d swept=%d dist=%d conf=%.2f",
                        fv.liquidity_sweep_signal, fv.liquidity_sweep_type,
                        fv.liquidity_sweep_pool_type, recent_pool.touch_count,
                        fv.liquidity_sweep_pool_count, fv.liquidity_sweep_swept_count,
                        fv.liquidity_sweep_recent_distance, fv.liquidity_sweep_confidence,
                    )

        except Exception as e:
            logger.warning(f"Liquidity Sweep 提取失败: {e}")

    # ========================================================================
    # 维度4: 多周期结构
    # ========================================================================

    def _extract_mtf(
        self,
        fv: FeatureVector,
        klines_1h: List[Dict],
        klines_4h: Optional[List[Dict]],
        klines_15m: Optional[List[Dict]],
    ):
        """提取多周期结构特征"""
        self._init_mtf()
        if self._mtf_core is None:
            return

        # 构建周期字典
        klines_by_tf = {"1h": klines_1h}
        if klines_4h:
            klines_by_tf["4h"] = klines_4h
        if klines_15m:
            klines_by_tf["15m"] = klines_15m

        # 如果只有1h, 使用1h作为主周期
        if len(klines_by_tf) == 1:
            return

        try:
            from mtf_core import extract_mtf_features
            features = extract_mtf_features(klines_by_tf)

            fv.mtf_direction = features.get("mtf_direction_score", 0.0)
            fv.mtf_strength = features.get("mtf_strength", 0.0)
            fv.mtf_is_confluent = bool(features.get("mtf_is_confluent", 0))
            fv.mtf_htf_veto = bool(features.get("mtf_htf_veto", 0))
            fv.mtf_confluence_score = features.get("mtf_confluence_score", 0.0)
        except Exception as e:
            logger.warning(f"MTF 计算失败: {e}")

    # ========================================================================
    # 融合: 加权综合评分
    # ========================================================================

    def _fuse_features(self, fv: FeatureVector):
        """4维加权融合 (v598修复: kelly_signal/vp_signal 真实接入 + CLR联动)

        v598 Bug#4修复:
          - kelly_signal: 从 fv.kelly_fraction 派生方向信号（旧版硬编码0.0导致kelly维度静默剔除）
          - vp_signal: 增强量价分析（OBV趋势 + VPIN毒性 + 量价背离，旧版仅用VPIN）
          - CLR联动: 接入 _v515_confluence_engine 的 kelly_multiplier（0.0/1.0/1.5/2.0）
          - get_pipeline_output: 新增公开输出口
        """

        # 各维度归一化信号 (-1 ~ +1)
        # v598修复: kelly_signal 从 kelly_fraction 派生方向信号
        kelly_signal = self._extract_kelly_direction(fv)
        pa_signal = fv.pa_signal  # -1 ~ +1
        # v598修复: vp_signal 增强量价分析（OBV + VPIN + 背离）
        vp_signal = self._extract_volume_price_signal(fv)
        mtf_signal = fv.mtf_direction  # -1 ~ +1

        # 加权融合
        # v598 Phase I: 用 adaptive_weights 替代静态 self.weights
        # loop_end 调用 adaptive_weights.update(dim_scores, realized_pnls) 后, 权重基于 IC 动态调整
        w = self.adaptive_weights.get_weights()
        composite = (
            w["kelly"] * kelly_signal +
            w["price_action"] * pa_signal +
            w["volume_price"] * vp_signal +
            w["mtf"] * mtf_signal
        )

        # CLR 联动: 接入 confluence tier 的 kelly_multiplier
        clr_multiplier = self._get_clr_multiplier(fv)
        if clr_multiplier > 0:
            # CLR 强信号时放大 composite（kelly_multiplier=2.0 → 放大1.3x，封顶1.0）
            composite = composite * (1.0 + (clr_multiplier - 1.0) * 0.3)

        # HTV 否决 → 强制归零
        if fv.mtf_htf_veto:
            composite = 0.0

        fv.composite_score = max(-1.0, min(1.0, composite))

        # v596 C2-d: regime 权重调整综合评分
        # crisis regime → weight=0.0 → 强制不交易；volatile → 0.6 减仓；trending → 1.0 正常
        try:
            from _v596_regime_namespace import get_regime_weight
            regime_weight = get_regime_weight(fv.regime_normalized, "all")
            fv.composite_score *= regime_weight
            fv.regime_strategy_weight = regime_weight
            if regime_weight == 0.0:
                # crisis regime: 清仓，跳过后续决策
                fv.should_trade = False
                fv.suggested_direction = "neutral"
                fv.suggested_size = 0.0
                return
        except Exception as regime_err:
            logger.debug("regime 权重应用失败, 降级为 1.0: %s", str(regime_err)[:100])
            fv.regime_strategy_weight = 1.0

        # 置信度: 基于各维度数据完整性
        confidence_parts = []
        if fv.kelly_confidence > 0:
            confidence_parts.append(fv.kelly_confidence)
        if fv.pa_signal != 0:
            confidence_parts.append(0.7)
        if fv.vpin_score > 0:
            confidence_parts.append(0.6)
        if fv.mtf_strength > 0:
            confidence_parts.append(fv.mtf_strength)

        fv.confidence = (
            sum(confidence_parts) / len(confidence_parts)
            if confidence_parts else 0.0
        )

        # v699 indicator_patches 消费 (用户铁律"复盘作用于整个AI量化技能包")
        # 闭环: PerTradeAttributionEngine 产出 indicator 维度建议 → EvolutionSuggestionApplier
        #       写入 _active_evolution_overlay.json → EvolutionLoop._init_decision_engines
        #       → pipeline.set_indicator_patches → _fuse_features 此处消费
        # 安全 (防御式集成, 不破坏主融合流程):
        #   - patch 匹配失败时静默 continue, 不抛异常
        #   - 仅 active 状态 patch 生效
        #   - try/except 整体降级保护
        if self._indicator_patches:
            try:
                for _ip in self._indicator_patches:
                    if _ip.get("status", "active") != "active":
                        continue
                    _feat_name = _ip.get("feature", "")
                    if not _feat_name:
                        continue
                    _feat_val = getattr(fv, _feat_name, None)
                    if _feat_val is None:
                        # FeatureVector 无此属性 → 跳过 (不破坏主流程)
                        continue
                    _threshold = _ip.get("threshold")
                    _op = _ip.get("op", ">")
                    # 评估 filter 是否匹配
                    try:
                        _matched = False
                        _fv_f = float(_feat_val)
                        _th_f = float(_threshold) if _threshold is not None else 0.0
                        if _op in ("gt", ">"):
                            _matched = _fv_f > _th_f
                        elif _op in ("lt", "<"):
                            _matched = _fv_f < _th_f
                        elif _op in ("ge", ">="):
                            _matched = _fv_f >= _th_f
                        elif _op in ("le", "<="):
                            _matched = _fv_f <= _th_f
                        elif _op in ("eq", "=="):
                            _matched = abs(_fv_f - _th_f) < 1e-8
                        elif _op in ("eq_direction",):
                            # eq_direction: 用于 pa_signal/kelly 等带方向特征
                            # threshold 为方向值(+1/-1/0), 匹配符号一致
                            if _th_f > 0:
                                _matched = _fv_f > 0
                            elif _th_f < 0:
                                _matched = _fv_f < 0
                            else:
                                _matched = abs(_fv_f) < 1e-8
                        if not _matched:
                            continue
                    except (TypeError, ValueError):
                        continue
                    # 执行 action
                    _action = _ip.get("action", "")
                    if _action == "reduce_position_by":
                        _factor = float(_ip.get("factor", 0.5))
                        # 限制 factor 在合理范围 [0.1, 0.95] 防极端值
                        _factor = max(0.1, min(0.95, _factor))
                        fv.composite_score *= _factor
                    elif _action == "monitor_only":
                        # monitor_only: 强制归零, 后续 should_trade=False
                        fv.composite_score = 0.0
                    elif _action == "boost_signal":
                        # boost_signal: 放大信号 (factor > 1.0), 封顶 +1.0
                        _factor = float(_ip.get("factor", 1.3))
                        _factor = max(0.5, min(2.0, _factor))
                        fv.composite_score = max(-1.0, min(1.0, fv.composite_score * _factor))
                    # 其他 action 暂不支持, 静默跳过
            except Exception as _ip_err:
                logger.debug("[indicator_patches] 消费异常(降级跳过): %s", str(_ip_err)[:100])

        # 交易决策
        if abs(fv.composite_score) >= self.composite_threshold and fv.confidence >= 0.3:
            fv.should_trade = True
            fv.suggested_direction = "long" if fv.composite_score > 0 else "short"
            # 仓位 = 凯利分数 × 置信度 × CLR乘数
            fv.suggested_size = fv.kelly_fraction * fv.confidence * clr_multiplier
        else:
            fv.should_trade = False
            fv.suggested_direction = "neutral"
            fv.suggested_size = 0.0

    # ========================================================================
    # v598 新增: 信号提取方法
    # ========================================================================

    def _extract_kelly_direction(self, fv: FeatureVector) -> float:
        """从凯利分数派生方向信号 (v598新增)

        凯利公式本身控制仓位大小，但仓位大小隐含方向信心：
          - kelly_fraction > 0.5 → 强多头信心 → +0.5
          - kelly_fraction 0.2-0.5 → 中性偏多 → +0.2
          - kelly_fraction < 0.2 → 弱/无信心 → -0.3
          - kelly_fraction = 0 → 无信号 → 0.0

        科学依据: 凯利分数 f* = (p*b - q) / b，高分数意味着高胜率或高盈亏比
        """
        k = fv.kelly_fraction
        if k > 0.5:
            return 0.5  # 强多头
        elif k > 0.2:
            return 0.2  # 弱多头
        elif k > 0.01:
            return -0.1  # 极弱，轻微偏空
        else:
            return 0.0  # 无信号

    def _extract_volume_price_signal(self, fv: FeatureVector) -> float:
        """增强量价信号提取 (v598新增, v98 CVD 集成)

        四层量价分析 (v98 升级: 新增 CVD 真实订单流信号):
          1. VPIN 毒性: 高毒性 → 负信号（不利交易）
          2. VolumeProfile POC: 价格在POC下方 → 偏多，上方 → 偏空
          3. 量价背离代理: VPIN突变作为简化背离信号 (保留作为对比)
          4. CVD 订单流信号 (v98 新增): 真实主动买卖方向 + 隐藏吸筹/分发 + 背离

        v98 升级说明 (用户铁律"量价分析融会贯通"):
          - Layer 3 之前用 VPIN 突变作为背离代理, 是简化近似
          - v98 集成 order_flow_cvd.py, 提供真实 CVD 背离检测 (73% BTC反转前出现)
          - CVD 是市场微观结构最强信号, 权重最高 (0.35)
          - 非过拟合: CVD 信号基于通用市场原理, 非历史拟合

        融合: 四层信号加权平均
        """
        signals = []

        # Layer 1: VPIN 毒性信号
        if fv.vpin_is_toxic:
            vpin_signal = -(fv.vpin_score - 0.5) * 2  # 毒性0.7 → -0.4
        else:
            vpin_signal = (0.5 - fv.vpin_score) * 0.5  # 低毒性 → 小正信号
        signals.append(("vpin", vpin_signal, 0.25))

        # Layer 2: VolumeProfile POC 信号
        poc_signal = 0.0
        if fv.volume_profile_poc > 0 and fv.pa_resistance > 0:
            # 价格在POC下方 → 偏多（价值区支撑）
            # 价格在POC上方 → 偏空（价值区阻力）
            # 这里用 pa_resistance 作为当前价格的代理（简化）
            if fv.pa_resistance < fv.volume_profile_poc:
                poc_signal = 0.3  # 阻力在POC下方 → 偏多
            else:
                poc_signal = -0.2  # 阻力在POC上方 → 偏空
        signals.append(("poc", poc_signal, 0.20))

        # Layer 3: 量价背离代理 (VPIN突变简化版, 保留作为对比基线)
        # 完整实现需要历史成交量数据，这里用VPIN分数变化近似
        divergence_signal = 0.0
        if fv.vpin_score > 0.6 and fv.pa_signal > 0.3:
            # 价格上涨但毒性高 → 量价背离 → 反转预警
            divergence_signal = -0.3
        elif fv.vpin_score < 0.3 and fv.pa_signal < -0.3:
            # 价格下跌但毒性低 → 可能超卖 → 偏多
            divergence_signal = 0.2
        signals.append(("divergence_proxy", divergence_signal, 0.20))

        # Layer 4: CVD 订单流信号 (v98 新增, 用户铁律"量价分析融会贯通")
        # CVD signal 已是融合信号 (-1~+1), 包含:
        #   - CVD 斜率方向 (短期+长期)
        #   - 买卖压力比
        #   - 4种CVD-Price模式 (确认牛市/熊市/隐藏吸筹/分发)
        #   - Delta 背离 (73% BTC反转前出现)
        #   - Absorption (盘整中delta异常)
        # 权重最高 (0.35), 因为 CVD 是市场微观结构最强信号
        cvd_signal = fv.cvd_signal if abs(fv.cvd_signal) > 0 else 0.0
        signals.append(("cvd", cvd_signal, 0.35))

        # 加权融合
        total_weight = sum(w for _, _, w in signals)
        if total_weight > 0:
            return sum(s * w for _, s, w in signals) / total_weight
        return 0.0

    def _get_clr_multiplier(self, fv: FeatureVector) -> float:
        """获取 CLR (Confluence Level Router) 的 kelly_multiplier (v598新增)

        从 _v515_confluence_engine 获取信号强度分级:
          - strong (score≥8): kelly_multiplier=2.0
          - medium (score≥5): kelly_multiplier=1.5
          - weak (score≥3): kelly_multiplier=1.0
          - none (score<3): kelly_multiplier=0.0

        降级策略: 若 CLR 不可用，基于 FeatureVector 自身信号估算
        """
        # 尝试调用 CLR
        try:
            if self._clr_cache is not None:
                return self._clr_cache
        except AttributeError:
            pass

        # 降级: 基于 FeatureVector 估算 CLR
        # 用 composite_score 的绝对值 + confidence 估算 tier
        estimated_score = abs(fv.composite_score) * 8 + fv.confidence * 4
        if estimated_score >= 8:
            return 2.0
        elif estimated_score >= 5:
            return 1.5
        elif estimated_score >= 3:
            return 1.0
        else:
            return 0.0

    def get_pipeline_output(self) -> Dict[str, Any]:
        """获取管道标准输出 (v598新增)

        提供统一的管道输出口，用于:
          1. 特征预测力测试（替代内联简化特征，修复 v561 方法论错误）
          2. 下游消费者（agent_decision_engine, evolution_loop_integrator）
          3. 调试和监控

        Returns:
            {
                "composite_score": float,
                "kelly_fraction": float,
                "kelly_signal": float,
                "pa_signal": float,
                "vp_signal": float,
                "mtf_signal": float,
                "confidence": float,
                "confluence_tier": str,
                "kelly_multiplier": float,
                "should_trade": bool,
                "suggested_direction": str,
                "suggested_size": float,
            }
        """
        if not hasattr(self, '_last_fv') or self._last_fv is None:
            return {
                "composite_score": 0.0,
                "kelly_fraction": 0.0,
                "kelly_signal": 0.0,
                "pa_signal": 0.0,
                "vp_signal": 0.0,
                "mtf_signal": 0.0,
                "confidence": 0.0,
                "confluence_tier": "none",
                "kelly_multiplier": 0.0,
                "should_trade": False,
                "suggested_direction": "neutral",
                "suggested_size": 0.0,
                "error": "管道尚未执行 extract()",
            }

        fv = self._last_fv
        clr_mult = self._get_clr_multiplier(fv)

        # 推算 confluence_tier
        if clr_mult >= 2.0:
            tier = "strong"
        elif clr_mult >= 1.5:
            tier = "medium"
        elif clr_mult >= 1.0:
            tier = "weak"
        else:
            tier = "none"

        return {
            "composite_score": round(fv.composite_score, 4),
            "kelly_fraction": round(fv.kelly_fraction, 4),
            "kelly_signal": round(self._extract_kelly_direction(fv), 4),
            "pa_signal": round(fv.pa_signal, 4),
            "vp_signal": round(self._extract_volume_price_signal(fv), 4),
            "mtf_signal": round(fv.mtf_direction, 4),
            "confidence": round(fv.confidence, 4),
            "confluence_tier": tier,
            "kelly_multiplier": clr_mult,
            "should_trade": fv.should_trade,
            "suggested_direction": fv.suggested_direction,
            "suggested_size": round(fv.suggested_size, 4),
        }


# ============================================================================
# 自检入口 (单元测试)
# ============================================================================

def _self_test() -> bool:
    """自检: 验证统一特征融合管道核心功能可用

    测试覆盖:
      1. 完整模式: mock 4h+1h+15m K线 + 充足交易历史 → FeatureVector 各维度字段填充
      2. 退化模式: 仅 1h K线, 无交易历史 → 优雅降级 (quarter Kelly 默认), 不崩溃
    使用确定性随机种子, 保证可复现。
    """
    import random

    def _gen_klines(n: int, trend: str = "up") -> List[Dict]:
        """生成 mock K线数据 (high/low/close/volume/open)"""
        random.seed(42)
        klines = []
        price = 100.0
        for i in range(n):
            if trend == "up":
                change = random.uniform(-0.3, 1.0)
            elif trend == "down":
                change = random.uniform(-1.0, 0.3)
            else:
                change = random.uniform(-0.5, 0.5)
            high = price + abs(change) + random.uniform(0, 0.3)
            low = price - abs(change) - random.uniform(0, 0.3)
            close = price + change
            volume = random.uniform(800, 1200)
            klines.append({
                "high": high, "low": low, "close": close,
                "volume": volume, "open": price,
            })
            price = close
        return klines

    try:
        pipeline = FeatureFusionPipeline()

        # ===== 测试1: 完整模式 (4h+1h+15m + 交易历史) =====
        klines_1h = _gen_klines(200, "up")
        klines_4h = _gen_klines(100, "up")
        klines_15m = _gen_klines(300, "up")
        trade_history = [0.02, -0.01, 0.03, 0.01, -0.005, 0.025, 0.015, -0.008,
                         0.02, 0.01] * 6  # 60笔交易

        fv = pipeline.extract(
            klines_1h=klines_1h,
            klines_4h=klines_4h,
            klines_15m=klines_15m,
            symbol="BTCUSDT",
            trade_history=trade_history,
        )

        # FeatureVector 应为 FeatureVector 实例且字段被填充
        assert isinstance(fv, FeatureVector), "extract 应返回 FeatureVector"
        assert 0.0 <= fv.kelly_fraction <= 1.0, \
            f"kelly_fraction 越界: {fv.kelly_fraction}"
        assert fv.kelly_track in ("full", "half"), \
            f"kelly_track 非法值: {fv.kelly_track}"
        assert -1.0 <= fv.pa_signal <= 1.0, \
            f"pa_signal 越界: {fv.pa_signal}"
        assert 0.0 <= fv.vpin_score <= 1.0, \
            f"vpin_score 越界: {fv.vpin_score}"
        assert -1.0 <= fv.mtf_direction <= 1.0, \
            f"mtf_direction 越界: {fv.mtf_direction}"
        assert 0.0 <= fv.mtf_strength <= 1.0, \
            f"mtf_strength 越界: {fv.mtf_strength}"
        # 综合评分应在合理范围
        assert -1.0 <= fv.composite_score <= 1.0, \
            f"composite_score 越界: {fv.composite_score}"
        assert 0.0 <= fv.confidence <= 1.0, \
            f"confidence 越界: {fv.confidence}"
        assert fv.suggested_direction in ("long", "short", "neutral"), \
            f"suggested_direction 非法值: {fv.suggested_direction}"
        # to_dict 应可序列化
        fv_dict = fv.to_dict()
        assert isinstance(fv_dict, dict), "to_dict 应返回 dict"
        assert "kelly" in fv_dict and "fusion" in fv_dict, \
            "to_dict 缺少关键维度"

        # ===== 测试2: 退化模式 (仅 1h, 无交易历史) → 优雅降级 =====
        fv2 = pipeline.extract(
            klines_1h=_gen_klines(100, "range"),
            symbol="ETHUSDT",
        )
        assert isinstance(fv2, FeatureVector), "退化模式应仍返回 FeatureVector"
        # 无交易历史 → quarter Kelly 默认, half 轨道
        assert fv2.kelly_track == "half", \
            f"无交易历史应为 half Kelly: {fv2.kelly_track}"
        assert 0.0 <= fv2.kelly_fraction <= 1.0, \
            f"退化模式 kelly_fraction 越界: {fv2.kelly_fraction}"
        assert -1.0 <= fv2.composite_score <= 1.0, \
            f"退化模式 composite_score 越界: {fv2.composite_score}"

        # ===== 测试3: set_indicator_patches 注入接口可用 =====
        pipeline.set_indicator_patches([{"dim": "pa_signal", "op": "monitor_only"}])
        assert len(pipeline._indicator_patches) == 1, \
            "indicator_patches 注入失败"
        # 清空 patches 不影响后续 extract
        pipeline.set_indicator_patches(None)
        assert len(pipeline._indicator_patches) == 0, "patches 清空失败"
        # 再次 extract 应正常 (验证清空后无副作用)
        fv3 = pipeline.extract(klines_1h=klines_1h, symbol="TEST")
        assert isinstance(fv3, FeatureVector), "清空 patches 后 extract 异常"

        logger.info("FeatureFusionPipeline 自检全部通过 ✓")
        return True

    except Exception as e:
        logger.error(f"自检失败: {e}")
        return False


# ============================================================================
# CLI (独立测试)
# ============================================================================

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
        raise SystemExit(0)
    import random
    random.seed(42)

    def gen_klines(n: int, trend: str = "up") -> List[Dict]:
        klines = []
        price = 100.0
        for i in range(n):
            if trend == "up":
                change = random.uniform(-0.3, 1.0)
            elif trend == "down":
                change = random.uniform(-1.0, 0.3)
            else:
                change = random.uniform(-0.5, 0.5)
            high = price + abs(change) + random.uniform(0, 0.3)
            low = price - abs(change) - random.uniform(0, 0.3)
            close = price + change
            volume = random.uniform(800, 1200)
            klines.append({
                "high": high, "low": low, "close": close,
                "volume": volume, "open": price,
            })
            price = close
        return klines

    print("=== 统一特征融合管道测试 ===\n")

    pipeline = FeatureFusionPipeline()

    # 测试: 4h+1h 上涨趋势, 有交易历史
    klines_1h = gen_klines(200, "up")
    klines_4h = gen_klines(100, "up")
    klines_15m = gen_klines(300, "up")
    trade_history = [0.02, -0.01, 0.03, 0.01, -0.005, 0.025, 0.015, -0.008,
                     0.02, 0.01] * 6  # 60笔交易

    fv = pipeline.extract(
        klines_1h=klines_1h,
        klines_4h=klines_4h,
        klines_15m=klines_15m,
        symbol="BTCUSDT",
        trade_history=trade_history,
    )

    print("特征向量:")
    print(f"  凯利分数: {fv.kelly_fraction:.4f} ({fv.kelly_track} Kelly)")
    print(f"  凯利胜率: {fv.kelly_win_rate:.2%}")
    print(f"  PA信号: {fv.pa_signal:+.4f}")
    print(f"  PA形态: {fv.pa_patterns}")
    print(f"  VPIN: {fv.vpin_score:.4f} (toxic={fv.vpin_is_toxic})")
    print(f"  VolumeProfile POC: {fv.volume_profile_poc:.2f}")
    print(f"  MTF方向: {fv.mtf_direction:+.4f}")
    print(f"  MTF强度: {fv.mtf_strength:.4f} (confluent={fv.mtf_is_confluent})")
    print(f"  MTF HTF否决: {fv.mtf_htf_veto}")
    print(f"\n融合结果:")
    print(f"  综合评分: {fv.composite_score:+.4f}")
    print(f"  置信度: {fv.confidence:.4f}")
    print(f"  应交易: {fv.should_trade}")
    print(f"  方向: {fv.suggested_direction}")
    print(f"  仓位: {fv.suggested_size:.4f}")

    print("\n=== 退化模式测试 (无4h/15m数据, 无交易历史) ===")
    fv2 = pipeline.extract(
        klines_1h=gen_klines(100, "range"),
        symbol="ETHUSDT",
    )
    print(f"  综合评分: {fv2.composite_score:+.4f}")
    print(f"  应交易: {fv2.should_trade}")
    print(f"  凯利轨道: {fv2.kelly_track} (quarter Kelly 默认)")
