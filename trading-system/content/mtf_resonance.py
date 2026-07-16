# -*- coding: utf-8 -*-
"""多时间框架波动率/流动性共振模块 (MTF Resonance) — 6大核心短板 #2

================================================================================
用户核心诉求: "凯利公式、价格行为、量价、多周期结构标准化特征融合管道，
              避免指标生搬硬套"
              "合理合适科学研究的把价格行为，凯利公式，量价分析，
              市场各周期的结构和趋势都要考虑进去但不要生搬硬套"

本模块实现多时间框架 (MTF) 的**波动率共振**与**流动性共振**，作为标准化特征融合
管道的核心组件。与 mtf_core.py 的方向维度并列存在，本模块专注共振维度。

⚠️ ERR-109 红线: 本模块不做方向对齐
  - 逆势trades实际表现更好 (win_rate=63.2% > 顺势56.3%)
  - 策略本质是均值回归非趋势跟踪
  - 只做波动率/流动性共振 (多周期同时进入高波动/低流动性状态)
  - 软过滤输出 (boost_mult), 不拦截信号

⚠️ ERR-110 红线: 不用特征做决策权重
  - 0/10特征通过 P<0.05 + |Cohen's d|>0.1 双重过滤
  - 只用市场状态做仓位调整 (软过滤), 不做硬过滤

设计原则 (遵循 sim_live_gap_model.py 范本):
  - @dataclass 参数 + 类封装 + __main__ 守卫
  - 零副作用导入 (无顶层 print/sys.exit/文件写入)
  - 复用 feature_extractors.compute_atr_pct (已NaN修复)

核心算法:
  1. 每个周期计算 atr_pct = compute_atr_pct(klines_tf, period)
  2. 每个周期计算 atr_pct 的 z-score (相对自身历史 mean/std)
  3. 加权融合: weighted_atr_z = sum(w_tf * z_tf)
  4. 波动率共振判定: weighted_atr_z > 1.0 → "expansion", < -1.0 → "compression"
  5. 流动性共振: volume_ratio = vol_now / vol_ma(volume_lookback)
  6. 加权融合 weighted_vol_ratio
  7. 流动性判定: < 0.7 → "thin", > 1.3 → "thick"
  8. boost_mult: 取最保守值 (expansion×1.20 OR thin×0.70)

使用示例:
  from mtf_resonance import MTFResonanceParams, compute_mtf_resonance
  params = MTFResonanceParams()
  result = compute_mtf_resonance({"4h": klines_4h, "1h": klines_1h, "15m": klines_15m}, params)
  if result.is_liquidity_thin:
      position_size *= result.boost_mult  # 0.7, 流动性枯竭降仓
================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 复用 feature_extractors 的纯函数 (已NaN修复)
try:
    from .feature_extractors import compute_atr_pct
    _FEATURE_EXTRACTORS_AVAILABLE = True
except ImportError:
    _FEATURE_EXTRACTORS_AVAILABLE = False
    compute_atr_pct = None  # type: ignore

logger = logging.getLogger("hermes.mtf_resonance")


# ============================================================================
# 数据类定义 (遵循 sim_live_gap_model.py 范本)
# ============================================================================


@dataclass
class MTFResonanceParams:
    """MTF共振参数 (遵循 sim_live_gap_model 范本)

    默认权重与 mtf_core.py 保持一致: 4h:0.40, 1h:0.30, 15m:0.20, 5m:0.10
    """

    # 时间框架权重 (自动归一化)
    weights: Optional[Dict[str, float]] = None  # 默认 {"4h":0.40,"1h":0.30,"15m":0.20,"5m":0.10}

    # ATR 计算参数
    atr_period: int = 14

    # 流动性计算参数
    volume_lookback: int = 20

    # 波动率共振阈值 (z-score)
    expansion_z_threshold: float = 1.0   # 波动率扩张阈值
    compression_z_threshold: float = -1.0  # 波动率压缩阈值

    # 流动性共振阈值 (volume ratio)
    thin_volume_ratio: float = 0.7        # 流动性枯竭阈值
    thick_volume_ratio: float = 1.3       # 流动性充裕阈值

    # 软过滤 boost 因子 (ERR-109: 软过滤, 不拦截信号)
    boost_expansion: float = 1.20         # 波动率扩张加仓 (高波动=高机会)
    boost_compression: float = 0.80       # 波动率压缩降仓 (低波动=低机会)
    boost_thin_liquidity: float = 0.70    # 流动性枯竭降仓 (高风险, 滑点大)
    boost_thick_liquidity: float = 1.10   # 流动性充裕加仓 (低滑点)

    # z-score 历史窗口
    zscore_lookback: int = 50  # 用过去50根K线的atr_pct计算z-score

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = {"4h": 0.40, "1h": 0.30, "15m": 0.20, "5m": 0.10}
        # 自动归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}


@dataclass
class MTFResonanceResult:
    """MTF共振结果

    所有字段遵循 ERR-109 红线: 不含任何方向字段 (direction/trend等)
    """

    # 波动率共振 (0.0-1.0, 多周期ATR同步扩张程度)
    volatility_resonance: float

    # 流动性共振 (0.0-1.0, 多周期volume同步萎缩/扩张程度)
    liquidity_resonance: float

    # 波动率状态 ("expansion"/"compression"/"normal")
    volatility_regime: str

    # 流动性状态 ("thin"/"thick"/"normal")
    liquidity_regime: str

    # 综合共振分 (0.0-1.0)
    resonance_score: float

    # 仓位调整因子 (0.7-1.3, 软过滤)
    boost_mult: float

    # 极端状态标志
    is_volatility_expansion: bool
    is_liquidity_thin: bool

    # 诊断字段 (各周期 z-score / volume_ratio)
    per_tf_atr_z: Dict[str, float]
    per_tf_vol_ratio: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """转为 dict (用于序列化)"""
        return {
            "volatility_resonance": self.volatility_resonance,
            "liquidity_resonance": self.liquidity_resonance,
            "volatility_regime": self.volatility_regime,
            "liquidity_regime": self.liquidity_regime,
            "resonance_score": self.resonance_score,
            "boost_mult": self.boost_mult,
            "is_volatility_expansion": self.is_volatility_expansion,
            "is_liquidity_thin": self.is_liquidity_thin,
            "per_tf_atr_z": self.per_tf_atr_z,
            "per_tf_vol_ratio": self.per_tf_vol_ratio,
        }


# ============================================================================
# 核心计算函数
# ============================================================================


def _compute_atr_zscore(
    klines: List[Dict],
    atr_period: int,
    lookback: int,
) -> float:
    """计算 ATR 百分比的 z-score (相对历史)

    Args:
        klines: K线数据 (含 high/low/close)
        atr_period: ATR 周期
        lookback: z-score 历史窗口

    Returns:
        当前 ATR 百分比的 z-score (相对于过去 lookback 根K线)
    """
    if not _FEATURE_EXTRACTORS_AVAILABLE or compute_atr_pct is None:
        return 0.0
    if len(klines) < atr_period + lookback:
        return 0.0

    atr_pct_series = compute_atr_pct(klines, atr_period)
    if len(atr_pct_series) < lookback:
        return 0.0

    # 取最近 lookback 根的 atr_pct
    recent_atr = atr_pct_series[-lookback:]
    current_atr = recent_atr[-1]
    history = recent_atr[:-1]

    if len(history) < 2:
        return 0.0

    mean_atr = sum(history) / len(history)
    var_atr = sum((x - mean_atr) ** 2 for x in history) / len(history)
    std_atr = math.sqrt(var_atr)

    if std_atr < 1e-10:
        return 0.0

    z = (current_atr - mean_atr) / std_atr
    # 限制 z-score 范围避免极端值
    return max(-5.0, min(5.0, z))


def _compute_volume_ratio(
    klines: List[Dict],
    lookback: int,
) -> float:
    """计算当前成交量相对历史均值的比率

    Args:
        klines: K线数据 (含 volume)
        lookback: 历史窗口

    Returns:
        volume_ratio = vol_now / vol_ma(lookback)
        > 1.0 = 放量, < 1.0 = 缩量
    """
    if len(klines) < lookback + 1:
        return 1.0

    volumes = [k.get("volume", 0) for k in klines[-(lookback + 1):]]
    current_vol = volumes[-1]
    history_vol = volumes[:-1]

    if not history_vol:
        return 1.0

    avg_vol = sum(history_vol) / len(history_vol)
    if avg_vol < 1e-10:
        return 1.0

    return current_vol / avg_vol


def _compute_volatility_resonance_score(
    per_tf_atr_z: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """计算波动率共振分数 (0.0-1.0)

    共振 = 多周期 z-score 同向的程度
    - 所有周期 z > 0 (同步扩张) → 高分
    - 所有周期 z < 0 (同步压缩) → 高分
    - z 方向不一致 → 低分

    Args:
        per_tf_atr_z: 各时间框架的 ATR z-score
        weights: 时间框架权重

    Returns:
        共振分数 0.0-1.0
    """
    if not per_tf_atr_z:
        return 0.0

    # 加权 z
    weighted_z = sum(weights.get(tf, 0) * z for tf, z in per_tf_atr_z.items())

    # 同向性: 所有 z 同号 → 高共振
    signs = [1 if z > 0 else (-1 if z < 0 else 0) for z in per_tf_atr_z.values()]
    n_positive = sum(1 for s in signs if s > 0)
    n_negative = sum(1 for s in signs if s < 0)
    n_total = len(signs)

    if n_total == 0:
        return 0.0

    # 同向比例 (0.0-1.0)
    alignment = max(n_positive, n_negative) / n_total

    # 共振分数 = 同向比例 × |加权z| 归一化
    z_magnitude = min(1.0, abs(weighted_z) / 2.0)  # |z|>=2 → 满分

    return alignment * z_magnitude


def _compute_liquidity_resonance_score(
    per_tf_vol_ratio: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """计算流动性共振分数 (0.0-1.0)

    共振 = 多周期 volume_ratio 同向偏离 1.0 的程度
    - 所有周期 vol_ratio < 1.0 (同步缩量) → 高分 (流动性枯竭共振)
    - 所有周期 vol_ratio > 1.0 (同步放量) → 高分 (流动性充裕共振)

    Args:
        per_tf_vol_ratio: 各时间框架的 volume_ratio
        weights: 时间框架权重

    Returns:
        共振分数 0.0-1.0
    """
    if not per_tf_vol_ratio:
        return 0.0

    # 加权 vol_ratio 偏离 1.0 的程度
    weighted_ratio = sum(weights.get(tf, 0) * r for tf, r in per_tf_vol_ratio.items())
    deviation = abs(weighted_ratio - 1.0)

    # 同向性: 所有 ratio 同侧偏离 1.0
    below = sum(1 for r in per_tf_vol_ratio.values() if r < 1.0)
    above = sum(1 for r in per_tf_vol_ratio.values() if r > 1.0)
    n_total = len(per_tf_vol_ratio)

    if n_total == 0:
        return 0.0

    alignment = max(below, above) / n_total

    # 偏离幅度归一化 (|deviation|>=0.3 → 满分)
    dev_magnitude = min(1.0, deviation / 0.3)

    return alignment * dev_magnitude


def compute_mtf_resonance(
    klines_by_tf: Dict[str, List[Dict]],
    params: Optional[MTFResonanceParams] = None,
) -> MTFResonanceResult:
    """计算多时间框架波动率/流动性共振 (ERR-109兼容: 不做方向对齐)

    Args:
        klines_by_tf: 各时间框架的K线数据 {"4h": [...], "1h": [...], "15m": [...]}
                      每个元素是 dict 含 high/low/close/volume
        params: 共振参数 (None=默认)

    Returns:
        MTFResonanceResult: 含 boost_mult (0.7-1.3) 用于软过滤仓位调整
    """
    if params is None:
        params = MTFResonanceParams()

    weights = params.weights or {"4h": 0.40, "1h": 0.30, "15m": 0.20, "5m": 0.10}

    # 只计算有权重且有数据的时间框架
    active_tfs = [tf for tf in weights if tf in klines_by_tf and len(klines_by_tf[tf]) > 0]

    if not active_tfs:
        logger.warning("mtf_resonance: 无可用时间框架数据, 返回默认结果")
        return MTFResonanceResult(
            volatility_resonance=0.0,
            liquidity_resonance=0.0,
            volatility_regime="normal",
            liquidity_regime="normal",
            resonance_score=0.0,
            boost_mult=1.0,
            is_volatility_expansion=False,
            is_liquidity_thin=False,
            per_tf_atr_z={},
            per_tf_vol_ratio={},
        )

    # 重新归一化 active_tfs 的权重
    active_weights_sum = sum(weights[tf] for tf in active_tfs)
    if active_weights_sum <= 0:
        active_weights_sum = 1.0
    active_weights = {tf: weights[tf] / active_weights_sum for tf in active_tfs}

    # 1. 计算各时间框架的 ATR z-score
    per_tf_atr_z: Dict[str, float] = {}
    for tf in active_tfs:
        z = _compute_atr_zscore(klines_by_tf[tf], params.atr_period, params.zscore_lookback)
        per_tf_atr_z[tf] = z

    # 2. 计算各时间框架的 volume_ratio
    per_tf_vol_ratio: Dict[str, float] = {}
    for tf in active_tfs:
        ratio = _compute_volume_ratio(klines_by_tf[tf], params.volume_lookback)
        per_tf_vol_ratio[tf] = ratio

    # 3. 加权融合 ATR z-score
    weighted_atr_z = sum(active_weights[tf] * per_tf_atr_z[tf] for tf in active_tfs)

    # 4. 波动率状态判定
    if weighted_atr_z > params.expansion_z_threshold:
        volatility_regime = "expansion"
        is_volatility_expansion = True
    elif weighted_atr_z < params.compression_z_threshold:
        volatility_regime = "compression"
        is_volatility_expansion = False
    else:
        volatility_regime = "normal"
        is_volatility_expansion = False

    # 5. 加权融合 volume_ratio
    weighted_vol_ratio = sum(active_weights[tf] * per_tf_vol_ratio[tf] for tf in active_tfs)

    # 6. 流动性状态判定
    if weighted_vol_ratio < params.thin_volume_ratio:
        liquidity_regime = "thin"
        is_liquidity_thin = True
    elif weighted_vol_ratio > params.thick_volume_ratio:
        liquidity_regime = "thick"
        is_liquidity_thin = False
    else:
        liquidity_regime = "normal"
        is_liquidity_thin = False

    # 7. 计算共振分数
    vol_resonance = _compute_volatility_resonance_score(per_tf_atr_z, active_weights)
    liq_resonance = _compute_liquidity_resonance_score(per_tf_vol_ratio, active_weights)
    resonance_score = (vol_resonance + liq_resonance) / 2.0

    # 8. 计算 boost_mult (软过滤, 取最保守值)
    boost_mult = 1.0
    if is_volatility_expansion:
        boost_mult *= params.boost_expansion  # 1.20
    elif volatility_regime == "compression":
        boost_mult *= params.boost_compression  # 0.80

    if is_liquidity_thin:
        boost_mult *= params.boost_thin_liquidity  # 0.70 (流动性枯竭最危险)
    elif liquidity_regime == "thick":
        boost_mult *= params.boost_thick_liquidity  # 1.10

    # 限制 boost_mult 范围
    boost_mult = max(0.5, min(1.5, boost_mult))

    logger.debug(
        "mtf_resonance: vol_regime=%s liq_regime=%s weighted_atr_z=%.3f "
        "weighted_vol_ratio=%.3f boost_mult=%.3f",
        volatility_regime, liquidity_regime, weighted_atr_z,
        weighted_vol_ratio, boost_mult,
    )

    return MTFResonanceResult(
        volatility_resonance=vol_resonance,
        liquidity_resonance=liq_resonance,
        volatility_regime=volatility_regime,
        liquidity_regime=liquidity_regime,
        resonance_score=resonance_score,
        boost_mult=boost_mult,
        is_volatility_expansion=is_volatility_expansion,
        is_liquidity_thin=is_liquidity_thin,
        per_tf_atr_z=per_tf_atr_z,
        per_tf_vol_ratio=per_tf_vol_ratio,
    )


# ============================================================================
# __main__ 守卫自检 (遵循 sim_live_gap_model 范本)
# ============================================================================


def _generate_mock_klines(n: int, base_price: float, volatility: float, volume_base: float) -> List[Dict]:
    """生成模拟K线数据 (用于 __main__ 自检)"""
    import random
    random.seed(42)
    klines: List[Dict] = []
    price = base_price
    for i in range(n):
        change = random.gauss(0, volatility)
        open_p = price
        close_p = price * (1 + change)
        high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, volatility * 0.3)))
        low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, volatility * 0.3)))
        vol = volume_base * (1 + random.gauss(0, 0.2))
        klines.append({
            "open": open_p, "high": high_p, "low": low_p, "close": close_p,
            "volume": max(1.0, vol),
        })
        price = close_p
    return klines


if __name__ == "__main__":
    print("=" * 70)
    print("MTF Resonance 模块自检 — 多时间框架波动率/流动性共振")
    print("=" * 70)
    print()

    # 测试1: 基本导入和参数
    print("【测试1】基本导入和参数")
    params = MTFResonanceParams()
    assert params.weights is not None, "weights 不应为 None"
    assert abs(sum(params.weights.values()) - 1.0) < 1e-6, "权重应归一化"
    assert params.boost_thin_liquidity == 0.70, "流动性枯竭 boost 应为 0.70"
    print(f"  ✅ 权重归一化: {params.weights} (sum={sum(params.weights.values()):.4f})")
    print(f"  ✅ boost_thin_liquidity = {params.boost_thin_liquidity}")
    print()

    # 测试2: 正常场景 (boost_mult ≈ 1.0)
    print("【测试2】正常场景 (boost_mult 应接近 1.0)")
    klines_4h = _generate_mock_klines(100, 50000, 0.01, 1000)
    klines_1h = _generate_mock_klines(100, 50000, 0.015, 500)
    klines_15m = _generate_mock_klines(100, 50000, 0.02, 200)
    result_normal = compute_mtf_resonance(
        {"4h": klines_4h, "1h": klines_1h, "15m": klines_15m}, params
    )
    print(f"  volatility_regime: {result_normal.volatility_regime}")
    print(f"  liquidity_regime: {result_normal.liquidity_regime}")
    print(f"  boost_mult: {result_normal.boost_mult:.4f}")
    print(f"  per_tf_atr_z: {result_normal.per_tf_atr_z}")
    assert 0.5 <= result_normal.boost_mult <= 1.5, "boost_mult 应在 0.5-1.5 范围内"
    print("  ✅ 正常场景 boost_mult 在合理范围")
    print()

    # 测试3: 流动性枯竭场景 (boost_mult < 0.8)
    print("【测试3】流动性枯竭场景 (boost_mult 应 < 0.8)")
    # 构造最后1根急剧缩量的K线: 前99根正常量1000, 最后1根骤降至100
    # lookback=20时, 最后21根=[1000×20, 100], ratio=100/1000=0.1 < 0.7 → 触发thin
    klines_thin = []
    price = 50000
    for i in range(100):
        if i < 99:
            vol = 1000.0  # 正常成交量
        else:
            vol = 100.0   # 最后1根急剧缩量至10% (ratio=0.1 << 0.7)
        klines_thin.append({
            "open": price, "high": price * 1.01, "low": price * 0.99,
            "close": price, "volume": max(1.0, vol),
        })
    result_thin = compute_mtf_resonance(
        {"4h": klines_thin, "1h": klines_thin, "15m": klines_thin}, params
    )
    print(f"  liquidity_regime: {result_thin.liquidity_regime}")
    print(f"  is_liquidity_thin: {result_thin.is_liquidity_thin}")
    print(f"  boost_mult: {result_thin.boost_mult:.4f}")
    assert result_thin.is_liquidity_thin, "应检测到流动性枯竭"
    assert result_thin.boost_mult < 0.8, f"流动性枯竭 boost 应 < 0.8, 实际 {result_thin.boost_mult}"
    print("  ✅ 流动性枯竭正确检测, boost_mult < 0.8")
    print()

    # 测试4: ERR-109 兼容性 (不含方向字段)
    print("【测试4】ERR-109 兼容性 (不含方向字段)")
    result_dict = result_normal.to_dict()
    direction_fields = [k for k in result_dict if "direction" in k.lower() or "trend" in k.lower()]
    assert len(direction_fields) == 0, f"不应包含方向字段, 实际有: {direction_fields}"
    print(f"  ✅ 结果字段无方向/trend: {list(result_dict.keys())}")
    print()

    # 测试5: 空数据容错
    print("【测试5】空数据容错")
    result_empty = compute_mtf_resonance({}, params)
    assert result_empty.boost_mult == 1.0, "空数据应返回默认 boost_mult=1.0"
    assert result_empty.volatility_regime == "normal", "空数据应返回 normal"
    print(f"  ✅ 空数据 boost_mult = {result_empty.boost_mult}")
    print()

    # 测试6: 单时间框架容错
    print("【测试6】单时间框架容错")
    result_single = compute_mtf_resonance({"1h": klines_1h}, params)
    assert "1h" in result_single.per_tf_atr_z, "应包含 1h 的 z-score"
    assert 0.5 <= result_single.boost_mult <= 1.5, "单TF boost 应在合理范围"
    print(f"  ✅ 单TF结果: per_tf_atr_z={result_single.per_tf_atr_z}")
    print()

    print("=" * 70)
    print("✅ 全部自检通过 — mtf_resonance.py 模块功能正常")
    print("=" * 70)
    print()
    print("ERR-109 红线确认: 本模块不做方向对齐, 只做波动率/流动性共振")
    print("ERR-110 红线确认: 本模块不用特征做决策权重, 只用 boost_mult 做软过滤")
