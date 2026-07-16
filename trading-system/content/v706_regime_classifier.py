# -*- coding: utf-8 -*-
"""v706 Regime Classifier — 6状态市场状态分类器

用户铁律:
  "在未达成性能指标前不得推进实盘部署工作"
  "永远永远不要出现模拟牛逼, 实盘亏损的情况"
  "不要通过已知数据反推策略, 那样就过拟合了!"

本模块封装 v704 Fix3 (regime 命名空间修复), 在 v706 OOS 环境中提供统一的
6状态 regime 识别接口, 供 V704SignalGenerator 使用:

架构 (三段式):
  1. 基础识别: _v531.classify_regime (5状态 EMA50/EMA200 检测)
     - 输出: strong_up / weak_up / strong_down / weak_down / ranging
  2. 命名归一化: _v596.normalize_regime (6状态标准枚举)
     - 输出: RegimeEnum.TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE / CRISIS / ACCUMULATION
  3. ATR 增强: 基于 ATR% 覆盖极端波动场景 (与 v704 KellyCriterionSizer 对齐)
     - ATR% >= 0.05  → CRISIS (conf=1.0, 极端波动)
     - ATR% >= 0.025 → VOLATILE (conf=0.7, 高波动)
     - ATR% < 0.005 AND regime==RANGING → ACCUMULATION (conf=0.6, 低波动蓄势)
     - 默认 conf=0.8

设计决策 (反过拟合):
  - ATR 阈值 0.05/0.025/0.005 来自 v97 v75.V72_FRICTION_PARAMS 的市场微观结构常识,
    非回测反推. 与 KellyCriterionSizer.REGIME_KELLY_STRATEGY 中 volatile/crisis 阈值对齐.
  - 置信度 0.6/0.7/0.8/1.0 为合理经验值, 非 grid-search 优化结果.
  - 5状态→6状态映射通过 REGIME_ALIASES 完成, 不引入新的命名系统.

来源:
  - _v531_v76_ltc_ranging_filter.py:81 — classify_regime
  - _v596_regime_namespace.py:41 — RegimeEnum
  - _v596_regime_namespace.py:218 — normalize_regime
  - _v534_v91_v596_integrated.py:332 — KellyCriterionSizer.get_kelly_fraction (regime阈值对齐)
  - v706_config.py:V706_RISK_PARAMS — atr_high_volatility=0.025, atr_medium_volatility=0.015
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

# v704 Fix3: 统一 regime 命名空间 (来自 _v596_regime_namespace)
from _v596_regime_namespace import RegimeEnum, normalize_regime

# v531: 5状态 EMA 检测器 (来自 _v531_v76_ltc_ranging_filter)
from _v531_v76_ltc_ranging_filter import classify_regime as _classify_regime_v531

logger = logging.getLogger("hermes.v706.regime_classifier")

# v99 Task #93 修复1: ADX+HH/HL 趋势确认 (避免EMA滞后误判, 基于Wilder 1978 + 道氏理论)
try:
    from _v99_task93_adx_structure_filter import enhance_regime_with_adx_structure as _enhance_with_adx
    _ADX_STRUCTURE_FILTER_AVAILABLE = True
    logger.info("v99 Task #93 ADX+structure filter 已加载 (修复1: BTC regime方向误判)")
except ImportError as _e:
    _ADX_STRUCTURE_FILTER_AVAILABLE = False
    logger.warning("v99 Task #93 ADX+structure filter 不可用: %s", str(_e)[:80])

# ============================================================================
# ATR 增强阈值 (与 v706_config.V706_RISK_PARAMS / KellyCriterionSizer 对齐)
# ============================================================================

# ATR% 极端波动阈值 (与 v706_config.V706_RISK_PARAMS.atr_high_volatility 对齐)
ATR_CRISIS_THRESHOLD = 0.05      # ATR% >= 5% → CRISIS (极端波动, 清仓)
ATR_VOLATILE_THRESHOLD = 0.025   # ATR% >= 2.5% → VOLATILE (高波动, 1/3 Kelly)
ATR_ACCUMULATION_THRESHOLD = 0.005  # ATR% < 0.5% → 可能蓄势

# 置信度默认值 (反过拟合: 经验值, 非 grid-search 优化)
CONF_DEFAULT = 0.8              # 默认置信度 (regime 识别可信)
CONF_CRISIS = 1.0               # CRISIS 强制覆盖 (极端波动确定性高)
CONF_VOLATILE = 0.7             # VOLATILE 覆盖 (高波动但未必危机)
CONF_ACCUMULATION = 0.6          # ACCUMULATION 覆盖 (低波动蓄势, 把握度较低)


class V706RegimeClassifier:
    """v706 6状态 regime 分类器 (封装 v531 + v596 + ATR增强)

    架构 (三段式):
      1. 基础识别: v531.classify_regime (5状态 EMA50/EMA200)
      2. 命名归一化: v596.normalize_regime (5状态 → 6状态 RegimeEnum)
      3. ATR 增强: 基于 ATR% 覆盖极端波动场景

    Attributes:
        _distribution: Dict[str, int] — 各 regime 出现次数统计 (调试+告警用)
        _total_calls: int — classify() 调用总数
        _atr_overrides: int — ATR 增强覆盖次数
    """

    def __init__(self) -> None:
        """初始化 v706 regime 分类器"""
        # regime 分布统计 (调试 + 告警用)
        self._distribution: Dict[str, int] = {r.value: 0 for r in RegimeEnum}
        self._total_calls: int = 0
        self._atr_overrides: int = 0

        logger.info(
            "V706RegimeClassifier initialized "
            "(ATR crisis>=%.3f, volatile>=%.3f, accumulation<=%.3f)",
            ATR_CRISIS_THRESHOLD, ATR_VOLATILE_THRESHOLD, ATR_ACCUMULATION_THRESHOLD,
        )

    # ========================================================================
    # 主入口: classify
    # ========================================================================

    def classify(
        self,
        klines: List[Dict[str, Any]],
        idx: int,
        atr_pct: Optional[float] = None,
    ) -> Tuple[RegimeEnum, float, str]:
        """识别 idx 处的 regime (6状态)

        三段式流程:
          1. v531.classify_regime(klines, idx) → raw_name (5状态 EMA)
          2. v596.normalize_regime(raw_name) → regime_enum (6状态)
          3. ATR 增强: 若 atr_pct 非空, 按 ATR% 阈值覆盖极端场景

        Args:
            klines: K线数据列表 (list of dict, 含 close/high/low)
            idx: 当前 K线索引 (需 >= 200, 否则返回 RANGING)
            atr_pct: 可选 ATR% (相对close的比例). None 时不做 ATR 增强.

        Returns:
            Tuple[RegimeEnum, float, str]:
                - regime_enum: 6状态标准枚举
                - confidence: 0.0-1.0, regime 识别置信度
                - raw_name: v531 原始 5状态命名 (调试用)

        Examples:
            >>> clf = V706RegimeClassifier()
            >>> regime, conf, raw = clf.classify(klines, 250, atr_pct=0.06)
            >>> regime
            <RegimeEnum.CRISIS: 'crisis'>
            >>> conf
            1.0
            >>> raw
            'ranging'
        """
        self._total_calls += 1

        # ===== 防御: 输入校验 =====
        if not klines or idx < 0:
            logger.warning("classify 输入异常: klines=%d, idx=%d", len(klines) if klines else 0, idx)
            return self._record(RegimeEnum.RANGING, 0.5, "unknown")

        if idx >= len(klines):
            logger.warning("classify idx 越界: idx=%d, len(klines)=%d", idx, len(klines))
            return self._record(RegimeEnum.RANGING, 0.5, "out_of_range")

        # ===== 步骤1: v531 基础识别 (5状态 EMA50/EMA200) =====
        try:
            raw_name = _classify_regime_v531(klines, idx)
        except Exception as e:
            logger.warning("v531 classify_regime 异常: %s, 默认 RANGING", str(e)[:80])
            raw_name = "ranging"

        # ===== 步骤2: v596 命名归一化 (5状态 → 6状态 RegimeEnum) =====
        try:
            regime_enum = normalize_regime(raw_name)
        except Exception as e:
            logger.warning("normalize_regime 异常: %s, 默认 RANGING", str(e)[:80])
            regime_enum = RegimeEnum.RANGING

        # ===== 步骤2.5: v99 Task #93 修复1 — ADX+HH/HL 趋势确认 =====
        # 目标: 避免EMA50/EMA200滞后导致的trending_down/up误判
        # 理论依据: Welles Wilder (1978) ADX + 道氏理论 HH/HL (非过拟合)
        # 规则:
        #   - EMA显示trending_down + ADX<20 → ranging (无明确趋势)
        #   - EMA显示trending_down + 价格已转HH/HL → ranging (EMA滞后)
        #   - EMA显示trending_up + ADX<20 → ranging
        #   - EMA显示trending_up + 价格已转LL/LH → ranging
        if _ADX_STRUCTURE_FILTER_AVAILABLE:
            try:
                enhanced_value, enhance_reason = _enhance_with_adx(
                    regime_enum.value, klines, idx)
                if enhanced_value != regime_enum.value:
                    # 转换回 RegimeEnum
                    try:
                        regime_enum = RegimeEnum(enhanced_value)
                    except ValueError:
                        pass  # 保持原值
                    logger.debug("v99 Task #93 增强: %s (idx=%d, raw=%s)",
                                enhance_reason, idx, raw_name)
            except Exception as e:
                logger.warning("ADX+structure增强异常 idx=%d: %s, 保持原regime=%s",
                              idx, str(e)[:80], regime_enum.value)

        # 默认置信度
        confidence = CONF_DEFAULT

        # ===== 步骤3: ATR 增强 (覆盖极端波动场景) =====
        if atr_pct is not None:
            try:
                atr_pct_f = float(atr_pct)
                # 防御: NaN/Inf
                import math as _math
                if _math.isnan(atr_pct_f) or _math.isinf(atr_pct_f):
                    atr_pct_f = 0.0
                atr_pct_f = max(0.0, atr_pct_f)
            except (TypeError, ValueError):
                atr_pct_f = 0.0

            # ATR% >= 5% → CRISIS (强制覆盖)
            if atr_pct_f >= ATR_CRISIS_THRESHOLD:
                regime_enum = RegimeEnum.CRISIS
                confidence = CONF_CRISIS
                self._atr_overrides += 1
            # ATR% >= 2.5% → VOLATILE (覆盖, 但不覆盖 trending_down/bear 状态)
            elif atr_pct_f >= ATR_VOLATILE_THRESHOLD:
                # 保留 trending_down (熊市可能也高波动, 但策略应做空而非清仓)
                if regime_enum != RegimeEnum.TRENDING_DOWN:
                    regime_enum = RegimeEnum.VOLATILE
                    confidence = CONF_VOLATILE
                    self._atr_overrides += 1
            # ATR% < 0.5% AND regime==RANGING → ACCUMULATION (低波动蓄势)
            elif atr_pct_f < ATR_ACCUMULATION_THRESHOLD and regime_enum == RegimeEnum.RANGING:
                regime_enum = RegimeEnum.ACCUMULATION
                confidence = CONF_ACCUMULATION
                self._atr_overrides += 1

        return self._record(regime_enum, confidence, raw_name)

    # ========================================================================
    # 统计 & 调试接口
    # ========================================================================

    def get_distribution(self) -> Dict[str, int]:
        """获取 regime 分布统计 (调试 + 告警用)

        Returns:
            Dict[str, int]: {regime_value: count}
                - "trending_up": 出现次数
                - "trending_down": 出现次数
                - ... (6个 regime)
                - "total": 总调用次数
                - "atr_overrides": ATR 增强覆盖次数
        """
        dist = dict(self._distribution)
        dist["total"] = self._total_calls
        dist["atr_overrides"] = self._atr_overrides
        return dist

    def get_regime_ratios(self) -> Dict[str, float]:
        """获取 regime 分布比例 (0.0-1.0)

        用于告警检查 (regime_single_state_max_ratio 阈值)
        """
        if self._total_calls == 0:
            return {r.value: 0.0 for r in RegimeEnum}
        return {
            r.value: self._distribution.get(r.value, 0) / self._total_calls
            for r in RegimeEnum
        }

    def reset_distribution(self) -> None:
        """重置分布统计 (用于新一轮 OOS 收集)"""
        self._distribution = {r.value: 0 for r in RegimeEnum}
        self._total_calls = 0
        self._atr_overrides = 0
        logger.info("V706RegimeClassifier distribution reset")

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _record(
        self,
        regime_enum: RegimeEnum,
        confidence: float,
        raw_name: str,
    ) -> Tuple[RegimeEnum, float, str]:
        """记录 regime 分布统计并返回结果"""
        key = regime_enum.value
        self._distribution[key] = self._distribution.get(key, 0) + 1
        return (regime_enum, confidence, raw_name)


# ============================================================================
# 自验证 (模块加载时自动运行, 确保 v531/v596 依赖可用)
# ============================================================================


def _self_validate() -> bool:
    """自验证: 确保依赖模块可调用且返回值符合预期"""
    try:
        # 验证 v531.classify_regime 可调用
        # 构造最小测试数据 (idx<200 应返回 "ranging")
        test_klines = [{"close": 100.0, "high": 101.0, "low": 99.0} for _ in range(10)]
        raw = _classify_regime_v531(test_klines, 5)
        if raw not in ("strong_up", "weak_up", "strong_down", "weak_down", "ranging"):
            logger.error("v531.classify_regime 返回值异常: %s", raw)
            return False

        # 验证 v596.normalize_regime 可调用
        enum_val = normalize_regime("bull")
        if enum_val != RegimeEnum.TRENDING_UP:
            logger.error("v596.normalize_regime 返回值异常: %s", enum_val)
            return False

        # 验证 ATR 阈值合理
        if ATR_CRISIS_THRESHOLD <= ATR_VOLATILE_THRESHOLD:
            logger.error("ATR 阈值异常: crisis=%.3f <= volatile=%.3f",
                         ATR_CRISIS_THRESHOLD, ATR_VOLATILE_THRESHOLD)
            return False

        return True
    except Exception as e:
        logger.error("V706RegimeClassifier 自验证失败: %s", str(e)[:100])
        return False


# 模块加载时自验证
assert _self_validate(), "V706RegimeClassifier 自验证失败, 检查 _v531/_v596 依赖"


# ============================================================================
# 快速自测 (python v706_regime_classifier.py)
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("v706 Regime Classifier — 自测")
    print("=" * 70)

    clf = V706RegimeClassifier()

    # 测试1: idx<200 → ranging
    klines_short = [{"close": 100.0, "high": 101.0, "low": 99.0} for _ in range(50)]
    regime, conf, raw = clf.classify(klines_short, 10, atr_pct=None)
    print(f"\n[Test 1] idx<200 (数据不足):")
    print(f"  regime={regime.value}, conf={conf}, raw={raw}")
    assert regime == RegimeEnum.RANGING, f"期望 RANGING, 实际 {regime}"
    print("  ✅ PASS")

    # 测试2: ATR%>=5% → CRISIS 覆盖
    klines_long = [{"close": 100.0, "high": 101.0, "low": 99.0} for _ in range(250)]
    regime, conf, raw = clf.classify(klines_long, 249, atr_pct=0.06)
    print(f"\n[Test 2] ATR%=6% (极端波动):")
    print(f"  regime={regime.value}, conf={conf}, raw={raw}")
    assert regime == RegimeEnum.CRISIS, f"期望 CRISIS, 实际 {regime}"
    assert conf == 1.0, f"期望 conf=1.0, 实际 {conf}"
    print("  ✅ PASS")

    # 测试3: ATR%>=2.5% → VOLATILE 覆盖
    regime, conf, raw = clf.classify(klines_long, 249, atr_pct=0.03)
    print(f"\n[Test 3] ATR%=3% (高波动):")
    print(f"  regime={regime.value}, conf={conf}, raw={raw}")
    assert regime == RegimeEnum.VOLATILE, f"期望 VOLATILE, 实际 {regime}"
    assert conf == 0.7, f"期望 conf=0.7, 实际 {conf}"
    print("  ✅ PASS")

    # 测试4: 分布统计
    dist = clf.get_distribution()
    print(f"\n[Test 4] 分布统计:")
    print(f"  {dist}")
    assert dist["total"] == 3, f"期望 total=3, 实际 {dist['total']}"
    assert dist["atr_overrides"] == 2, f"期望 atr_overrides=2, 实际 {dist['atr_overrides']}"
    print("  ✅ PASS")

    # 测试5: 重置分布
    clf.reset_distribution()
    dist = clf.get_distribution()
    assert dist["total"] == 0, f"期望 total=0, 实际 {dist['total']}"
    print(f"\n[Test 5] 重置分布: ✅ PASS")

    print("\n" + "=" * 70)
    print("✅ v706 Regime Classifier 自测全部通过")
    print("=" * 70)
