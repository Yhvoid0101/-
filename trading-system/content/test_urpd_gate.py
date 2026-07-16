#!/usr/bin/env python3
"""URPD/UTXO实现价格门控模块测试 — test_urpd_gate.py

测试覆盖:
  - 枚举值验证
  - 数据结构 defaults & to_dict
  - URPDGate 初始化
  - _classify_level 全等级覆盖
  - analyze 正常/边界/异常
  - apply_to_decision 全门控规则覆盖 (8种组合)
  - get_stats / reset / round_complete
  - 缓存读写 roundtrip
  - MVRV 计算正确性
"""
from __future__ import annotations

import json
import os
import sys

import pytest

# 确保能从当前目录导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from urpd_realized_price import (
    URPDAction,
    URPDAnalysisResult,
    URPDGate,
    URPDLevel,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_cache():
    """每个测试前后清理缓存文件，防止测试间互相干扰"""
    cache_file = URPDGate.CACHE_FILE
    if os.path.exists(cache_file):
        os.remove(cache_file)
    yield
    if os.path.exists(cache_file):
        os.remove(cache_file)


def make_gate(realized_price=None):
    """创建URPDGate实例，可选mock fetch_from_api返回值

    Args:
        realized_price: 若为数值，fetch_from_api返回该值; 若为None，返回None(走缓存/估算)

    Returns:
        URPDGate 实例
    """
    gate = URPDGate()
    gate.fetch_from_api = lambda: realized_price
    return gate


# ============================================================================
# 1. 枚举测试
# ============================================================================

class TestURPDEnums:
    def test_urpd_level_values(self):
        """URPDLevel 枚举值验证"""
        assert URPDLevel.DEEP_DISCOUNT.value == "deep_discount"
        assert URPDLevel.DISCOUNT.value == "discount"
        assert URPDLevel.NEUTRAL.value == "neutral"
        assert URPDLevel.PREMIUM.value == "premium"
        assert URPDLevel.EXTREME_PREMIUM.value == "extreme_premium"

    def test_urpd_action_values(self):
        """URPDAction 枚举值验证"""
        assert URPDAction.ALLOW.value == "allow"
        assert URPDAction.BOOST.value == "boost"
        assert URPDAction.REDUCE.value == "reduce"
        assert URPDAction.NEUTRAL.value == "neutral"


# ============================================================================
# 2. 数据结构测试
# ============================================================================

class TestURPDAnalysisResult:
    def test_defaults(self):
        """URPDAnalysisResult 默认值"""
        r = URPDAnalysisResult()
        assert r.market_price == 0.0
        assert r.realized_price == 0.0
        assert r.mvrv_ratio == 1.0
        assert r.level == URPDLevel.NEUTRAL
        assert r.has_data is False

    def test_to_dict(self):
        """URPDAnalysisResult.to_dict() 返回正确字段"""
        r = URPDAnalysisResult(
            market_price=65000.0,
            realized_price=45500.0,
            mvrv_ratio=1.428,
            level=URPDLevel.NEUTRAL,
            has_data=True,
        )
        d = r.to_dict()
        assert d["market_price"] == 65000.0
        assert d["realized_price"] == 45500.0
        assert d["mvrv_ratio"] == pytest.approx(1.428)
        assert d["level"] == "neutral"
        assert d["has_data"] is True
        assert "timestamp" in d


# ============================================================================
# 3. 初始化测试
# ============================================================================

class TestURPDGateInit:
    def test_init_enabled(self):
        """初始化 enabled=True"""
        gate = URPDGate(enabled=True)
        assert gate.enabled is True
        stats = gate.get_stats()
        assert stats["total_calls"] == 0
        assert stats["adjust_rate"] == 0.0

    def test_init_disabled(self):
        """初始化 enabled=False"""
        gate = URPDGate(enabled=False)
        assert gate.enabled is False


# ============================================================================
# 4. _classify_level 测试 (全等级覆盖)
# ============================================================================

class TestClassifyLevel:
    def test_deep_discount(self):
        """MVRV < 0.5 → DEEP_DISCOUNT"""
        gate = URPDGate()
        assert gate._classify_level(0.1) == URPDLevel.DEEP_DISCOUNT
        assert gate._classify_level(0.3) == URPDLevel.DEEP_DISCOUNT
        assert gate._classify_level(0.49) == URPDLevel.DEEP_DISCOUNT

    def test_discount(self):
        """0.5 <= MVRV < 0.8 → DISCOUNT"""
        gate = URPDGate()
        assert gate._classify_level(0.5) == URPDLevel.DISCOUNT
        assert gate._classify_level(0.6) == URPDLevel.DISCOUNT
        assert gate._classify_level(0.79) == URPDLevel.DISCOUNT

    def test_neutral(self):
        """0.8 <= MVRV <= 1.5 → NEUTRAL"""
        gate = URPDGate()
        assert gate._classify_level(0.8) == URPDLevel.NEUTRAL
        assert gate._classify_level(1.0) == URPDLevel.NEUTRAL
        assert gate._classify_level(1.5) == URPDLevel.NEUTRAL

    def test_premium(self):
        """1.5 < MVRV <= 2.5 → PREMIUM"""
        gate = URPDGate()
        assert gate._classify_level(1.51) == URPDLevel.PREMIUM
        assert gate._classify_level(2.0) == URPDLevel.PREMIUM
        assert gate._classify_level(2.5) == URPDLevel.PREMIUM

    def test_extreme_premium(self):
        """MVRV > 2.5 → EXTREME_PREMIUM"""
        gate = URPDGate()
        assert gate._classify_level(2.51) == URPDLevel.EXTREME_PREMIUM
        assert gate._classify_level(3.0) == URPDLevel.EXTREME_PREMIUM
        assert gate._classify_level(5.0) == URPDLevel.EXTREME_PREMIUM


# ============================================================================
# 5. analyze 测试
# ============================================================================

class TestAnalyze:
    def test_analyze_valid_price(self):
        """analyze 正常市场价格"""
        gate = make_gate(realized_price=45500.0)
        result = gate.analyze(65000.0)
        assert result.has_data is True
        assert result.market_price == 65000.0
        assert result.realized_price == 45500.0
        assert result.mvrv_ratio == pytest.approx(65000.0 / 45500.0)

    def test_analyze_zero_price(self):
        """analyze 市场价为0 → 无数据"""
        gate = URPDGate()
        result = gate.analyze(0.0)
        assert result.has_data is False
        assert result.market_price == 0.0

    def test_analyze_negative_price(self):
        """analyze 负市场价格 → 无数据"""
        gate = URPDGate()
        result = gate.analyze(-100.0)
        assert result.has_data is False

    def test_analyze_estimation_fallback(self):
        """API和缓存都失败时，使用估算 realized = market * 0.7"""
        gate = make_gate(realized_price=None)
        result = gate.analyze(100.0)
        assert result.has_data is True
        assert result.realized_price == pytest.approx(70.0)
        assert result.mvrv_ratio == pytest.approx(100.0 / 70.0)


# ============================================================================
# 6. apply_to_decision 测试 (全门控规则覆盖)
# ============================================================================

class TestApplyToDecision:
    def test_disabled_returns_original(self):
        """enabled=False 时返回原decision（同一对象）"""
        gate = URPDGate(enabled=False)
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result is decision

    def test_market_price_zero_no_data(self):
        """market_price=0 时记为no_data，不写门控字段"""
        gate = URPDGate()
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 0.0)
        assert "urpd_gate_action" not in result
        assert gate.get_stats()["no_data_count"] == 1

    # ---- 深度折价 (MVRV<0.5) ----
    def test_deep_discount_long_boost(self):
        """深度折价 + 做多 → BOOST ×1.25"""
        gate = make_gate(realized_price=250.0)  # MVRV = 100/250 = 0.4
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "boost"
        assert result["urpd_gate_multiplier"] == pytest.approx(1.25)
        assert result["quantity"] == pytest.approx(1.25)
        assert result["urpd_mvrv_ratio"] == pytest.approx(0.4)

    def test_deep_discount_short_reduce(self):
        """深度折价 + 做空 → REDUCE ×0.5"""
        gate = make_gate(realized_price=250.0)
        decision = {"action": "short", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "reduce"
        assert result["urpd_gate_multiplier"] == pytest.approx(0.5)
        assert result["quantity"] == pytest.approx(0.5)

    # ---- 折价 (MVRV<0.8) ----
    def test_discount_long_boost(self):
        """折价 + 做多 → BOOST ×1.15"""
        gate = make_gate(realized_price=130.0)  # MVRV ≈ 0.769
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "boost"
        assert result["urpd_gate_multiplier"] == pytest.approx(1.15)
        assert result["quantity"] == pytest.approx(1.15)

    def test_discount_short_reduce(self):
        """折价 + 做空 → REDUCE ×0.7"""
        gate = make_gate(realized_price=130.0)
        decision = {"action": "short", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "reduce"
        assert result["urpd_gate_multiplier"] == pytest.approx(0.7)
        assert result["quantity"] == pytest.approx(0.7)

    # ---- 极端溢价 (MVRV>2.5) ----
    def test_extreme_premium_long_reduce(self):
        """极端溢价 + 做多 → REDUCE ×0.5"""
        gate = make_gate(realized_price=30.0)  # MVRV ≈ 3.333
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "reduce"
        assert result["urpd_gate_multiplier"] == pytest.approx(0.5)
        assert result["quantity"] == pytest.approx(0.5)

    def test_extreme_premium_short_boost(self):
        """极端溢价 + 做空 → BOOST ×1.25"""
        gate = make_gate(realized_price=30.0)
        decision = {"action": "short", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "boost"
        assert result["urpd_gate_multiplier"] == pytest.approx(1.25)
        assert result["quantity"] == pytest.approx(1.25)

    # ---- 溢价 (MVRV>1.5) ----
    def test_premium_long_reduce(self):
        """溢价 + 做多 → REDUCE ×0.7"""
        gate = make_gate(realized_price=55.0)  # MVRV ≈ 1.818
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "reduce"
        assert result["urpd_gate_multiplier"] == pytest.approx(0.7)
        assert result["quantity"] == pytest.approx(0.7)

    def test_premium_short_boost(self):
        """溢价 + 做空 → BOOST ×1.15"""
        gate = make_gate(realized_price=55.0)
        decision = {"action": "short", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "boost"
        assert result["urpd_gate_multiplier"] == pytest.approx(1.15)
        assert result["quantity"] == pytest.approx(1.15)

    # ---- 中性 (0.8<=MVRV<=1.5) ----
    def test_neutral_allow(self):
        """中性 → ALLOW，仓位不变"""
        gate = make_gate(realized_price=80.0)  # MVRV = 1.25
        decision = {"action": "long", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "allow"
        assert result["urpd_gate_multiplier"] == pytest.approx(1.0)
        assert result["quantity"] == pytest.approx(1.0)

    # ---- action 关键词覆盖 ----
    def test_action_buy_keyword(self):
        """action='buy' 识别为做多"""
        gate = make_gate(realized_price=250.0)  # DEEP_DISCOUNT
        decision = {"action": "buy", "quantity": 2.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "boost"

    def test_action_sell_keyword(self):
        """action='sell' 识别为做空"""
        gate = make_gate(realized_price=250.0)
        decision = {"action": "sell", "quantity": 2.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "reduce"

    def test_action_flat_no_adjustment(self):
        """action='flat' 不触发任何调整 → ALLOW"""
        gate = make_gate(realized_price=30.0)  # EXTREME_PREMIUM
        decision = {"action": "flat", "quantity": 1.0, "price": 100}
        result = gate.apply_to_decision(decision, "BTC-USDT", 100.0)
        assert result["urpd_gate_action"] == "allow"
        assert result["urpd_gate_multiplier"] == pytest.approx(1.0)


# ============================================================================
# 7. 统计与回调测试
# ============================================================================

class TestStatsAndCallbacks:
    def test_get_stats_fields(self):
        """get_stats 返回所有必要字段"""
        gate = URPDGate()
        stats = gate.get_stats()
        for key in ("enabled", "total_calls", "total_adjusted",
                     "boost_count", "reduce_count", "no_data_count",
                     "adjust_rate", "last_analysis",
                     "realized_price_history_len"):
            assert key in stats, f"missing key: {key}"

    def test_stats_after_calls(self):
        """多次调用后统计正确"""
        gate = make_gate(realized_price=250.0)  # DEEP_DISCOUNT
        decision = {"action": "long", "quantity": 1.0}
        gate.apply_to_decision(decision, "BTC", 100.0)
        gate.apply_to_decision(decision, "BTC", 100.0)
        stats = gate.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_adjusted"] == 2
        assert stats["boost_count"] == 2
        assert stats["reduce_count"] == 0
        assert stats["adjust_rate"] == pytest.approx(1.0)

    def test_reset(self):
        """reset 清空所有统计"""
        gate = make_gate(realized_price=250.0)
        decision = {"action": "long", "quantity": 1.0}
        gate.apply_to_decision(decision, "BTC", 100.0)
        assert gate.get_stats()["total_calls"] > 0
        gate.reset()
        stats = gate.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_adjusted"] == 0
        assert stats["boost_count"] == 0
        assert stats["reduce_count"] == 0

    def test_round_complete(self):
        """round_complete 不抛出异常"""
        gate = URPDGate()
        gate.round_complete()  # 不应报错


# ============================================================================
# 8. 缓存测试
# ============================================================================

class TestCache:
    def test_cache_roundtrip(self):
        """缓存写入和读取 roundtrip"""
        gate = URPDGate()
        gate._save_to_cache(45500.0)
        loaded = gate.load_from_cache()
        assert loaded is not None
        assert loaded == pytest.approx(45500.0)

    def test_cache_not_found(self):
        """缓存文件不存在时返回None"""
        gate = URPDGate()
        result = gate.load_from_cache()
        assert result is None

    def test_estimation_saves_to_cache(self):
        """估算时会自动保存到缓存"""
        gate = make_gate(realized_price=None)
        gate.analyze(100.0)
        assert os.path.exists(URPDGate.CACHE_FILE)
        with open(URPDGate.CACHE_FILE, "r") as f:
            data = json.load(f)
        assert data["realized_price"] == pytest.approx(70.0)


# ============================================================================
# 9. MVRV 计算测试
# ============================================================================

class TestMVRVCalculation:
    def test_mvrv_calculation(self):
        """MVRV = market_price / realized_price"""
        gate = make_gate(realized_price=50000.0)
        result = gate.analyze(100000.0)
        assert result.mvrv_ratio == pytest.approx(2.0)
        assert result.level == URPDLevel.PREMIUM

    def test_mvrv_below_1_means_loss(self):
        """MVRV < 1 表示持有者亏损"""
        gate = make_gate(realized_price=100000.0)
        result = gate.analyze(40000.0)
        assert result.mvrv_ratio == pytest.approx(0.4)
        assert result.level == URPDLevel.DEEP_DISCOUNT

    def test_mvrv_above_1_means_profit(self):
        """MVRV > 1 表示持有者盈利"""
        gate = make_gate(realized_price=50000.0)
        result = gate.analyze(75000.0)
        assert result.mvrv_ratio == pytest.approx(1.5)
        assert result.level == URPDLevel.NEUTRAL
