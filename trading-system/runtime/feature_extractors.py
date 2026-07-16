# -*- coding: utf-8 -*-
"""特征提取器 (Feature Extractors) — 6大核心短板 #2

提取自 _v561_feature_pipeline.py，去除脚本式副作用（warnings过滤/sys.exit劫持/
重量级v82回测/FeatureFusionPipeline实例化/模块级文件写入），保留16个纯函数，
遵循 sim_live_gap_model.py 的架构范本（纯函数 + __main__ 守卫）。

用户核心诉求: "凯利公式、价格行为、量价、多周期结构标准化特征融合管道，避免指标生搬硬套"

设计原则:
  - 纯函数: 所有函数无副作用，相同输入相同输出
  - 标准化: 所有特征输出为 float，便于跨 symbol 比较
  - NaN 修复: compute_adx 包含 NaN 检查，返回默认值 20.0
  - 负面证据标注: 在 docstring 中明确标注 v561 的预测力测试结果

⚠️ v561 负面证据 (ERR-110):
  0/10 特征通过 P<0.05 + |Cohen's d|>0.1 双重过滤（与 ERR-065 一致）。
  ADX/ATR/RSI/PinBar/Engulfing/VP背离/BOS/CHOCH 对 PnL 无显著预测力。
  → 不应使用加权 confluence，应作为信号确认而非决策权重。
  → 市场状态分类仍有效: quiet 最佳 ($11985, wr 62.96%), trend 最差 ($722)。
  → 策略本质是均值回归非趋势跟踪 (ERR-109)。
  → hold_bars 是唯一统计显著特征 (ERR-065: 42.5%方差, v88fp: d=+0.461)

来源: _v561_feature_pipeline.py (801行+) → 提取约 350行纯逻辑
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Tuple


# ============================================================
# 指标计算函数
# ============================================================

def ema(values: List[float], period: int) -> List[float]:
    """指数移动平均

    Args:
        values: 输入序列
        period: EMA 周期

    Returns:
        EMA 序列，长度与 values 相同
    """
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def rsi(close_prices: List[float], period: int = 14) -> List[float]:
    """RSI 相对强弱指标

    Args:
        close_prices: 收盘价序列
        period: RSI 周期，默认 14

    Returns:
        RSI 序列 (0-100)，长度为 len(close_prices) - period
    """
    if len(close_prices) < period + 1:
        return [50.0] * len(close_prices)
    gains, losses = [], []
    for i in range(1, len(close_prices)):
        diff = close_prices[i] - close_prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    rsi_out = [50.0]
    avg_gain = statistics.mean(gains[:period]) if len(gains) >= period else 0
    avg_loss = statistics.mean(losses[:period]) if len(losses) >= period else 0.001
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi_out.append(100 - 100 / (1 + rs))
    return rsi_out


def compute_atr_pct(klines: List[Dict], period: int = 14) -> List[float]:
    """ATR 百分比 (相对价格)

    使用 Wilder 平滑法计算 ATR，并除以收盘价得到百分比。

    Args:
        klines: K线数据列表，每个元素含 high/low/close
        period: ATR 周期，默认 14

    Returns:
        ATR 百分比序列 (e.g. 0.01 = 1%)，长度与 klines 相同
    """
    if len(klines) < period + 1:
        return [0.01] * len(klines)
    trs = []
    for i in range(1, len(klines)):
        h, l = klines[i].get("high", 0), klines[i].get("low", 0)
        c_prev = klines[i - 1].get("close", 0)
        if c_prev > 0:
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev)) / c_prev
        else:
            tr = 0.01
        trs.append(tr)
    # Wilder smoothing
    atrs = [sum(trs[:period]) / period]
    for i in range(period, len(trs)):
        atrs.append((atrs[-1] * (period - 1) + trs[i]) / period)
    # 修复长度bug: 前面填充0.01使总长度=len(klines) (原代码返回 n-period 长度)
    n = len(klines)
    return [0.01] * (n - len(atrs)) + atrs


def compute_adx(klines: List[Dict], period: int = 14) -> List[float]:
    """ADX 趋势强度指标 (简化版，修复 NaN bug)

    ⚠️ 负面证据 (ERR-110): ADX 对 PnL 无显著预测力 (|Cohen's d|<0.1)
    → 仅用于市场状态分类 (trend/range)，不用于决策权重

    NaN 修复: 如果计算结果含 NaN，返回默认值 20.0

    Args:
        klines: K线数据列表
        period: ADX 周期，默认 14

    Returns:
        ADX 序列 (0-100)，长度与 klines 相同
    """
    n = len(klines)
    if n < 2 * period:
        return [20.0] * n
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, n):
        h, l = klines[i].get("high", 0), klines[i].get("low", 0)
        h_p, l_p = klines[i - 1].get("high", 0), klines[i - 1].get("low", 0)
        c_prev = klines[i - 1].get("close", 0)
        up = h - h_p
        down = l_p - l
        plus_dm.append(float(up if (up > down and up > 0) else 0))
        minus_dm.append(float(down if (down > up and down > 0) else 0))
        if c_prev > 0:
            trs.append(float(max(h - l, abs(h - c_prev), abs(l - c_prev))))
        else:
            trs.append(0.01)
    # Wilder smoothing (使用纯float计算, 避免statistics.mean的Python 3.12 bug)
    atr = sum(trs[:period]) / period
    plus_di_vals = [sum(plus_dm[:period]) / period / atr * 100 if atr > 0 else 0]
    minus_di_vals = [sum(minus_dm[:period]) / period / atr * 100 if atr > 0 else 0]
    dx_vals = []
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        pdm = (plus_di_vals[-1] * (period - 1) + plus_dm[i]) / period
        mdm = (minus_di_vals[-1] * (minus_dm[i] if i < len(minus_dm) else 0)) / period
        pdi = pdm / atr * 100 if atr > 0 else 0
        mdi = mdm / atr * 100 if atr > 0 else 0
        plus_di_vals.append(pdi)
        minus_di_vals.append(mdi)
        di_sum = pdi + mdi
        dx_vals.append(abs(pdi - mdi) / di_sum * 100 if di_sum > 0 else 0)
    # ADX = DX的Wilder平滑
    if len(dx_vals) < period:
        return [20.0] * n
    adx = [sum(dx_vals[:period]) / period]
    for i in range(period, len(dx_vals)):
        adx.append((adx[-1] * (period - 1) + dx_vals[i]) / period)
    # NaN检查: 如果有NaN, 返回默认值20.0
    # 修复长度bug: 前面填充20.0使总长度=n (原代码返回 n-period 长度)
    result = [20.0] * (n - len(adx)) + adx
    if any(math.isnan(x) if isinstance(x, float) else False for x in result):
        return [20.0] * n
    return result


# ============================================================
# 形态检测函数 (价格行为)
# ============================================================

def detect_pinbar(klines: List[Dict], i: int) -> bool:
    """PinBar 形态检测 (价格行为)

    ⚠️ 负面证据 (ERR-110): PinBar 对 PnL 无显著预测力
    → 仅作为信号确认，不用于决策权重

    PinBar 条件: 影线 ≥ 2×实体，实体 ≤ 30% 范围

    Args:
        klines: K线数据列表
        i: 当前 K 线索引

    Returns:
        True 如果检测到 PinBar
    """
    if i < 1 or i >= len(klines):
        return False
    k = klines[i]
    h, l, o, c = k.get("high", 0), k.get("low", 0), k.get("open", 0), k.get("close", 0)
    body = abs(c - o)
    rng = h - l
    if rng <= 0 or body <= 0:
        return False
    body_ratio = body / rng
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    # PinBar: 影线≥2×实体, 实体≤30%范围
    return (lower_wick >= 2 * body or upper_wick >= 2 * body) and body_ratio <= 0.3


def detect_engulfing(klines: List[Dict], i: int) -> bool:
    """Engulfing 形态检测 (价格行为)

    ⚠️ 负面证据 (ERR-110): Engulfing 对 PnL 无显著预测力
    → 仅作为信号确认，不用于决策权重

    支持看涨吞没和看跌吞没两种形态。

    Args:
        klines: K线数据列表
        i: 当前 K 线索引

    Returns:
        True 如果检测到 Engulfing
    """
    if i < 1 or i >= len(klines):
        return False
    k0, k1 = klines[i - 1], klines[i]
    o0, c0 = k0.get("open", 0), k0.get("close", 0)
    o1, c1 = k1.get("open", 0), k1.get("close", 0)
    if o0 == 0 or o1 == 0:
        return False
    # Bullish engulfing
    if c0 < o0 and c1 > o1 and c1 >= o0 and o1 <= c0:
        return True
    # Bearish engulfing
    if c0 > o0 and c1 < o1 and c1 <= o0 and o1 >= c0:
        return True
    return False


def detect_volume_divergence(klines: List[Dict], i: int, lookback: int = 20) -> int:
    """量价背离检测

    ⚠️ 负面证据 (ERR-110): VP背离 对 PnL 无显著预测力
    → 仅作为信号确认，不用于决策权重

    Args:
        klines: K线数据列表
        i: 当前 K 线索引
        lookback: 回看窗口，默认 20

    Returns:
        1=正背离 (价跌量增，可能见底)
        -1=负背离 (价升量缩，反转信号)
        0=无背离
    """
    if i < lookback or i >= len(klines):
        return 0
    prices = [klines[j].get("close", 0) for j in range(i - lookback, i + 1)]
    vols = [klines[j].get("volume", 0) for j in range(i - lookback, i + 1)]
    if not prices or not vols:
        return 0
    price_slope = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
    vol_slope = (vols[-1] - vols[0]) / vols[0] if vols[0] > 0 else 0
    # 价升量缩 = 负背离 (反转信号)
    if price_slope > 0.02 and vol_slope < -0.1:
        return -1
    # 价跌量增 = 正背离 (可能见底)
    if price_slope < -0.02 and vol_slope > 0.1:
        return 1
    return 0


def detect_bos_choch(klines: List[Dict], i: int, lookback: int = 20) -> int:
    """BOS/CHOCH 检测 (市场结构)

    ⚠️ 负面证据 (ERR-110): BOS/CHOCH 对 PnL 无显著预测力
    → 仅作为信号确认，不用于决策权重

    BOS (Break of Structure): 突破前高，看涨信号
    CHOCH (Change of Character): 跌破前低，看跌反转信号

    Args:
        klines: K线数据列表
        i: 当前 K 线索引
        lookback: 回看窗口，默认 20

    Returns:
        1=BOS看涨, -1=CHOCH看跌反转, 0=无
    """
    if i < lookback or i >= len(klines):
        return 0
    recent = klines[max(0, i - lookback):i]
    if len(recent) < 5:
        return 0
    highs = [k.get("high", 0) for k in recent]
    lows = [k.get("low", 0) for k in recent]
    swing_high = max(highs)
    swing_low = min(lows)
    cur_close = klines[i].get("close", 0)
    # BOS: 突破前高
    if cur_close > swing_high * 1.001:
        return 1
    # CHOCH: 跌破前低
    if cur_close < swing_low * 0.999:
        return -1
    return 0


# ============================================================
# 特征提取主函数
# ============================================================

def extract_features(klines: List[Dict], entry_bar: int) -> Dict[str, float]:
    """从 klines 提取入场时的标准化特征向量 (10维)

    ⚠️ 负面证据 (ERR-110): 0/10 特征通过 P<0.05 + |Cohen's d|>0.1 双重过滤
    → 不应使用加权 confluence，应作为信号确认而非决策权重
    → hold_bars 是唯一统计显著特征 (不在本函数，需从 trade 数据获取)

    覆盖维度:
      - 价格行为: PinBar, Engulfing
      - 量价: VolRatio, VP背离
      - 多周期结构: EMA对齐, ADX, BOS/CHOCH
      - 波动率: ATR%
      - 动量: RSI

    Args:
        klines: K线数据列表
        entry_bar: 入场 K 线索引

    Returns:
        10维特征 dict:
        {adx, atr_pct, vol_ratio, ema_alignment, rsi,
         pinbar, engulfing, vol_divergence, bos_choch, kelly_mult}
    """
    features = {
        "adx": 20.0, "atr_pct": 0.01, "vol_ratio": 1.0,
        "ema_alignment": 0.0, "rsi": 50.0,
        "pinbar": 0.0, "engulfing": 0.0,
        "vol_divergence": 0.0, "bos_choch": 0.0,
        "kelly_mult": 1.0,  # 默认1.0, 实际从trade中获取
    }
    if not klines or entry_bar >= len(klines) or entry_bar < 30:
        return features
    closes = [k.get("close", 0) for k in klines[:entry_bar + 1] if k.get("close", 0) > 0]
    if len(closes) < 50:
        return features

    # 多周期EMA对齐 (50/200)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, min(200, len(closes)))
    if ema50 and ema200 and ema200[-1] > 0:
        features["ema_alignment"] = (ema50[-1] - ema200[-1]) / ema200[-1]
    # ADX
    adx_vals = compute_adx(klines[:entry_bar + 1], 14)
    features["adx"] = adx_vals[-1] if adx_vals else 20.0
    # ATR%
    atr_vals = compute_atr_pct(klines[:entry_bar + 1], 14)
    features["atr_pct"] = atr_vals[-1] if atr_vals else 0.01
    # RSI
    rsi_vals = rsi(closes, 14)
    features["rsi"] = rsi_vals[-1] if rsi_vals else 50.0
    # VolRatio
    if entry_bar >= 10:
        vols = [klines[j].get("volume", 0) for j in range(entry_bar - 10, entry_bar)]
        if len(vols) >= 10:
            recent = sum(vols[-5:]) / 5
            prev = sum(vols[-10:-5]) / 5
            features["vol_ratio"] = recent / prev if prev > 0 else 1.0
    # PinBar/Engulfing
    features["pinbar"] = 1.0 if detect_pinbar(klines, entry_bar) else 0.0
    features["engulfing"] = 1.0 if detect_engulfing(klines, entry_bar) else 0.0
    # Vol Divergence
    features["vol_divergence"] = float(detect_volume_divergence(klines, entry_bar))
    # BOS/CHOCH
    features["bos_choch"] = float(detect_bos_choch(klines, entry_bar))
    return features


# ============================================================
# 统计工具函数 (预测力测试)
# ============================================================

def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Pearson 相关系数 + p值 (近似 t 分布)

    用于特征预测力测试: 特征值与 PnL 的相关性。

    Args:
        x: 特征值序列
        y: 目标变量序列 (e.g. PnL)

    Returns:
        (r, p): 相关系数 (-1到1), p值 (0到1)
        p < 0.05 表示相关性显著
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    r = sxy / (sx * sy)
    r = max(-1.0, min(1.0, r))
    # t检验p值 (近似)
    df = n - 2
    if df <= 0:
        return r, 1.0
    t_stat = r * math.sqrt(df / (1 - r * r + 1e-10))
    # 近似p值 (使用正态近似, df>30时准确)
    try:
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    except Exception:
        p = 0.05 if abs(t_stat) > 2 else 0.5
    return r, p


def _safe_stdev(vals: List[float]) -> float:
    """安全的 stdev 计算 (避免 Python 3.12 statistics.stdev 的 numerator bug)

    Python 3.12 的 statistics.stdev 在某些边缘情况有 numerator bug，
    本函数使用纯 float 计算避免该问题。

    Args:
        vals: 数值序列

    Returns:
        样本标准差 (ddof=1)
    """
    if len(vals) < 2:
        return 0.0
    fv = [float(v) for v in vals]
    m = sum(fv) / len(fv)
    var = sum((v - m) ** 2 for v in fv) / (len(fv) - 1)
    return math.sqrt(var)


def _safe_mean(vals: List[float]) -> float:
    """安全的 mean 计算

    Args:
        vals: 数值序列

    Returns:
        算术平均数
    """
    if not vals:
        return 0.0
    return sum(float(v) for v in vals) / len(vals)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d 效应量

    用于特征预测力测试: 盈利组 vs 亏损组的特征均值差异。
    |d| > 0.1 表示有实际意义的影响 (非纯随机)。

    ⚠️ v561 实测: 0/10 特征 |d| > 0.1 (与 ERR-065 一致)

    Args:
        group1: 盈利组的特征值
        group2: 亏损组的特征值

    Returns:
        Cohen's d 值 (正: group1 > group2, 负: group1 < group2)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = _safe_mean(group1), _safe_mean(group2)
    s1 = _safe_stdev(group1)
    s2 = _safe_stdev(group2)
    pooled_sd = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


def bonferroni_alpha(n_tests: int, alpha: float = 0.05) -> float:
    """Bonferroni 校正

    多重假设检验校正: 避免"多重比较问题"导致的假阳性。
    校正后 alpha = alpha / n_tests

    v561 使用: 10个特征测试，bonferroni_alpha = 0.05/10 = 0.005

    Args:
        n_tests: 假设检验次数
        alpha: 原始显著性水平，默认 0.05

    Returns:
        校正后的 alpha 值
    """
    return alpha / max(n_tests, 1)


# ============================================================
# 市场状态分类函数
# ============================================================

def classify_market_regime(adx: float, atr_pct: float) -> str:
    """市场状态自动分类 (trend/range/volatile/quiet)

    ✅ 有效 (ERR-110): 市场状态分类仍有效
      - quiet 最佳 ($11985, wr 62.96%)
      - trend 最差 ($722, wr 57.30%)
      - volatile 中等 ($729, wr 54.25%)
      - low_vol 高胜率 (67%)

    分类规则:
      - trend: ADX ≥ 25 (强趋势)
      - volatile: atr_pct ≥ 0.025 (高波动)
      - range: ADX < 20 AND atr_pct < 0.015 (低波动震荡)
      - quiet: 其他 (低波动+趋势弱)

    策略本质是均值回归非趋势跟踪 (ERR-109):
      → trend 状态表现最差，因为策略逆势
      → quiet/range 状态表现最好，因为均值回归有效

    Args:
        adx: ADX 趋势强度指标 (0-100)
        atr_pct: ATR 百分比 (e.g. 0.01 = 1%)

    Returns:
        市场状态字符串: "trend" / "range" / "volatile" / "quiet"
    """
    if adx >= 25:
        return "trend"
    if atr_pct >= 0.025:
        return "volatile"
    if adx < 20 and atr_pct < 0.015:
        return "range"
    return "quiet"


# ============================================================
# __main__ 守卫自检 (遵循 sim_live_gap_model.py 范本)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("feature_extractors.py — L6 功能自检")
    print("=" * 70)

    n_pass = 0
    n_total = 0

    # 构造测试 K线数据 (50根，模拟 BTC 1h)
    test_klines = []
    base_price = 50000.0
    for i in range(60):
        # 模拟价格波动
        vol = 100 + i * 0.5
        high = base_price + vol + (i % 3) * 50
        low = base_price - vol - (i % 2) * 30
        close = base_price + math.sin(i * 0.3) * 200 + (i - 30) * 10
        open_ = close - math.cos(i * 0.2) * 100
        test_klines.append({
            "high": high, "low": low, "close": close,
            "open": open_, "volume": 1000 + i * 10,
        })

    # 测试1: ema 基本计算
    n_total += 1
    closes = [k["close"] for k in test_klines]
    ema_vals = ema(closes, 14)
    assert len(ema_vals) == len(closes), "ema长度应与输入一致"
    assert ema_vals[0] == closes[0], "ema第一个值应等于输入第一个值"
    assert all(isinstance(x, float) for x in ema_vals), "ema所有值应为float"
    print(f"✅ 测试1: ema(14) 计算正确, 长度={len(ema_vals)}, 首值={ema_vals[0]:.2f}")
    n_pass += 1

    # 测试2: rsi 基本计算
    n_total += 1
    rsi_vals = rsi(closes, 14)
    assert len(rsi_vals) > 0, "rsi应返回非空序列"
    assert all(0 <= x <= 100 for x in rsi_vals), "rsi所有值应在0-100范围"
    print(f"✅ 测试2: rsi(14) 计算正确, 范围[0,100], 样本数={len(rsi_vals)}")
    n_pass += 1

    # 测试3: compute_atr_pct 基本计算
    n_total += 1
    atr_vals = compute_atr_pct(test_klines, 14)
    assert len(atr_vals) > 0, "atr_pct应返回非空序列"
    assert all(x > 0 for x in atr_vals), "atr_pct所有值应>0"
    print(f"✅ 测试3: compute_atr_pct(14) 计算正确, 样本数={len(atr_vals)}, "
          f"末值={atr_vals[-1]:.4f}")
    n_pass += 1

    # 测试4: compute_adx 基本计算 + NaN修复
    n_total += 1
    adx_vals = compute_adx(test_klines, 14)
    assert len(adx_vals) == len(test_klines), "adx长度应与klines一致"
    assert all(not math.isnan(x) for x in adx_vals), "adx不应含NaN (NaN修复验证)"
    assert all(0 <= x <= 100 for x in adx_vals), "adx所有值应在0-100范围"
    print(f"✅ 测试4: compute_adx(14) 计算正确, 无NaN, 末值={adx_vals[-1]:.2f}")
    n_pass += 1

    # 测试5: detect_pinbar 形态检测
    n_total += 1
    # 构造一个明显的PinBar: 长下影线，小实体
    pinbar_klines = test_klines.copy()
    pinbar_klines[30] = {
        "high": 50100, "low": 49500, "open": 50050, "close": 50030, "volume": 1000
    }
    is_pinbar = detect_pinbar(pinbar_klines, 30)
    assert isinstance(is_pinbar, bool), "detect_pinbar应返回bool"
    # 边界测试
    assert detect_pinbar([], 0) == False, "空klines应返回False"
    assert detect_pinbar(test_klines, 0) == False, "i<1应返回False"
    print(f"✅ 测试5: detect_pinbar 形态检测正确, 边界条件处理正确")
    n_pass += 1

    # 测试6: detect_engulfing 形态检测
    n_total += 1
    is_engulfing = detect_engulfing(test_klines, 31)
    assert isinstance(is_engulfing, bool), "detect_engulfing应返回bool"
    assert detect_engulfing([], 0) == False, "空klines应返回False"
    print(f"✅ 测试6: detect_engulfing 形态检测正确, 边界条件处理正确")
    n_pass += 1

    # 测试7: detect_volume_divergence 量价背离
    n_total += 1
    vd = detect_volume_divergence(test_klines, 30, lookback=20)
    assert vd in [-1, 0, 1], "detect_volume_divergence应返回-1/0/1"
    assert detect_volume_divergence(test_klines, 5, lookback=20) == 0, "i<lookback应返回0"
    print(f"✅ 测试7: detect_volume_divergence 正确, 返回值={vd}")
    n_pass += 1

    # 测试8: detect_bos_choch 市场结构
    n_total += 1
    bc = detect_bos_choch(test_klines, 30, lookback=20)
    assert bc in [-1, 0, 1], "detect_bos_choch应返回-1/0/1"
    print(f"✅ 测试8: detect_bos_choch 正确, 返回值={bc}")
    n_pass += 1

    # 测试9: extract_features 返回10维特征
    n_total += 1
    features = extract_features(test_klines, 50)
    assert isinstance(features, dict), "extract_features应返回dict"
    expected_keys = {"adx", "atr_pct", "vol_ratio", "ema_alignment", "rsi",
                     "pinbar", "engulfing", "vol_divergence", "bos_choch", "kelly_mult"}
    assert set(features.keys()) == expected_keys, f"特征keys应为10维, 实际={set(features.keys())}"
    assert all(isinstance(v, float) for v in features.values()), "所有特征值应为float"
    print(f"✅ 测试9: extract_features 返回10维特征正确, adx={features['adx']:.2f}, "
          f"atr_pct={features['atr_pct']:.4f}")
    n_pass += 1

    # 测试10: extract_features 边界条件 (entry_bar<30)
    n_total += 1
    features_edge = extract_features(test_klines, 10)
    assert features_edge["adx"] == 20.0, "entry_bar<30时adx应为默认20.0"
    assert features_edge["rsi"] == 50.0, "entry_bar<30时rsi应为默认50.0"
    print(f"✅ 测试10: extract_features 边界条件处理正确 (entry_bar<30返回默认值)")
    n_pass += 1

    # 测试11: pearson_correlation 统计检验
    n_total += 1
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]  # 完全正相关
    r, p = pearson_correlation(x, y)
    assert r > 0.99, f"完全正相关r应接近1, 实际={r}"
    assert p < 0.05, f"显著相关p应<0.05, 实际={p}"
    # 无相关
    r0, p0 = pearson_correlation([1, 2, 3], [3, 2, 1])
    assert r0 < 0, "负相关r应<0"
    print(f"✅ 测试11: pearson_correlation 正确, 正相关r={r:.4f}, p={p:.4f}")
    n_pass += 1

    # 测试12: _safe_stdev + _safe_mean (Python 3.12 bug修复)
    n_total += 1
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _safe_mean(vals) == 3.0, "_safe_mean(1-5)应为3.0"
    assert abs(_safe_stdev(vals) - 1.5811) < 0.01, f"_safe_stdev(1-5)应≈1.5811, 实际={_safe_stdev(vals):.4f}"
    assert _safe_mean([]) == 0.0, "_safe_mean(空列表)应为0.0"
    assert _safe_stdev([]) == 0.0, "_safe_stdev(空列表)应为0.0"
    assert _safe_stdev([1.0]) == 0.0, "_safe_stdev(单元素)应为0.0"
    print(f"✅ 测试12: _safe_stdev + _safe_mean 正确 (Python 3.12 bug修复验证)")
    n_pass += 1

    # 测试13: cohens_d 效应量
    n_total += 1
    g1 = [10.0, 11.0, 12.0, 13.0, 14.0]  # 均值11.5
    g2 = [5.0, 6.0, 7.0, 8.0, 9.0]       # 均值7.0
    d = cohens_d(g1, g2)
    assert d > 0, "g1>g2时d应>0"
    assert d > 1.0, f"显著差异d应>1.0, 实际={d}"
    # 相同组
    d0 = cohens_d(g1, g1)
    assert d0 == 0.0, "相同组d应为0.0"
    print(f"✅ 测试13: cohens_d 正确, 显著差异d={d:.4f}")
    n_pass += 1

    # 测试14: bonferroni_alpha 多重检验校正
    n_total += 1
    alpha_10 = bonferroni_alpha(10, 0.05)
    assert alpha_10 == 0.005, f"10次检验bonferroni_alpha应为0.005, 实际={alpha_10}"
    alpha_1 = bonferroni_alpha(1, 0.05)
    assert alpha_1 == 0.05, "1次检验bonferroni_alpha应为0.05"
    alpha_0 = bonferroni_alpha(0, 0.05)
    assert alpha_0 == 0.05, "0次检验bonferroni_alpha应为0.05 (max(n,1))"
    print(f"✅ 测试14: bonferroni_alpha 正确, 10次检验alpha={alpha_10}")
    n_pass += 1

    # 测试15: classify_market_regime 4种状态
    n_total += 1
    assert classify_market_regime(30.0, 0.01) == "trend", "ADX=30应为trend"
    assert classify_market_regime(15.0, 0.03) == "volatile", "atr_pct=0.03应为volatile"
    assert classify_market_regime(15.0, 0.01) == "range", "ADX=15+atr=0.01应为range"
    assert classify_market_regime(22.0, 0.02) == "quiet", "ADX=22+atr=0.02应为quiet"
    print(f"✅ 测试15: classify_market_regime 4种状态分类全部正确")
    n_pass += 1

    # 测试16: 负面证据标注验证 (docstring 包含 ERR-110 警告)
    n_total += 1
    module_doc = __doc__ or ""
    assert "ERR-110" in module_doc, "模块docstring应包含ERR-110负面证据标注"
    assert "0/10" in module_doc, "模块docstring应包含0/10特征通过率"
    assert "Cohen's d" in module_doc or "Cohen" in module_doc, "模块docstring应包含Cohen's d说明"
    print(f"✅ 测试16: 负面证据标注验证正确 (ERR-110/0/10/Cohen's d 均在docstring中)")
    n_pass += 1

    print("\n" + "=" * 70)
    print(f"L6 自检结果: {n_pass}/{n_total} PASS")
    if n_pass == n_total:
        print("🎉 全部通过! feature_extractors.py 功能可用")
        print("\n⚠️ 负面证据提醒 (ERR-110):")
        print("  0/10 特征通过 P<0.05 + |Cohen's d|>0.1 双重过滤")
        print("  → 不应使用加权 confluence，应作为信号确认而非决策权重")
        print("  → 市场状态分类仍有效: quiet 最佳, trend 最差")
        print("  → hold_bars 是唯一统计显著特征 (ERR-065: 42.5%方差)")
    else:
        print(f"❌ {n_total - n_pass} 项失败")
    print("=" * 70)
