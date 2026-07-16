#!/usr/bin/env python3
"""FearGreedGate 模块测试 — 恐慌贪婪指数门控

测试覆盖:
  1. 等级分类 (5个等级 + 边界值)
  2. 门控逻辑 (8种组合: 4等级 × 2方向)
  3. 中性 ALLOW
  4. disabled 开关
  5. 统计追踪
  6. 缓存加载 (无文件)
  7. round_complete 回调
  8. flat 动作
"""
from __future__ import annotations

import os
import sys
import pytest

# 确保能导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fear_greed_gate import (
    FearGreedGate,
    FGLevel,
    FGAction,
    FGAnalysisResult,
)


# ============================================================================
# Fixture
# ============================================================================

@pytest.fixture
def gate():
    """创建一个启用状态的 FearGreedGate"""
    return FearGreedGate(enabled=True)


@pytest.fixture
def disabled_gate():
    """创建一个禁用状态的 FearGreedGate"""
    return FearGreedGate(enabled=False)


def make_decision(action: str, quantity: float = 1.0, price: float = 50000.0) -> dict:
    """构造测试用决策字典"""
    return {"action": action, "quantity": quantity, "price": price}


# ============================================================================
# 测试1: 等级分类 — 极度恐慌
# ============================================================================

class TestLevelClassification:
    """测试恐慌贪婪指数等级分类"""

    def test_extreme_fear(self, gate):
        """index=10 → EXTREME_FEAR"""
        result = gate.analyze(index_value=10)
        assert result.level == FGLevel.EXTREME_FEAR
        assert result.has_data is True
        assert result.index_value == 10

    def test_fear(self, gate):
        """index=30 → FEAR"""
        result = gate.analyze(index_value=30)
        assert result.level == FGLevel.FEAR
        assert result.has_data is True

    def test_neutral(self, gate):
        """index=50 → NEUTRAL"""
        result = gate.analyze(index_value=50)
        assert result.level == FGLevel.NEUTRAL
        assert result.has_data is True

    def test_greed(self, gate):
        """index=70 → GREED"""
        result = gate.analyze(index_value=70)
        assert result.level == FGLevel.GREED
        assert result.has_data is True

    def test_extreme_greed(self, gate):
        """index=90 → EXTREME_GREED"""
        result = gate.analyze(index_value=90)
        assert result.level == FGLevel.EXTREME_GREED
        assert result.has_data is True


# ============================================================================
# 测试2: 边界值
# ============================================================================

class TestBoundaryValues:
    """测试阈值边界值"""

    def test_boundary_20_is_fear(self, gate):
        """index=20 → FEAR (不是 EXTREME_FEAR)"""
        result = gate.analyze(index_value=20)
        assert result.level == FGLevel.FEAR

    def test_boundary_40_is_neutral(self, gate):
        """index=40 → NEUTRAL"""
        result = gate.analyze(index_value=40)
        assert result.level == FGLevel.NEUTRAL

    def test_boundary_60_is_greed(self, gate):
        """index=60 → GREED"""
        result = gate.analyze(index_value=60)
        assert result.level == FGLevel.GREED

    def test_boundary_80_is_extreme_greed(self, gate):
        """index=80 → EXTREME_GREED"""
        result = gate.analyze(index_value=80)
        assert result.level == FGLevel.EXTREME_GREED

    def test_boundary_19_is_extreme_fear(self, gate):
        """index=19.9 → EXTREME_FEAR"""
        result = gate.analyze(index_value=19.9)
        assert result.level == FGLevel.EXTREME_FEAR


# ============================================================================
# 测试3: 门控逻辑 — 极度恐慌
# ============================================================================

class TestExtremeFearGating:
    """测试极度恐慌下的门控逻辑"""

    def test_extreme_fear_long_boost(self, gate):
        """极度恐慌 + 做多 → BOOST ×1.25"""
        decision = make_decision("long", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=10)
        assert result["fg_gate_action"] == FGAction.BOOST.value
        assert result["fg_gate_multiplier"] == 1.25
        assert result["quantity"] == pytest.approx(1.25)
        assert "fg_gate_reason" in result

    def test_extreme_fear_short_reduce(self, gate):
        """极度恐慌 + 做空 → REDUCE ×0.5"""
        decision = make_decision("short", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=15)
        assert result["fg_gate_action"] == FGAction.REDUCE.value
        assert result["fg_gate_multiplier"] == 0.5
        assert result["quantity"] == pytest.approx(0.5)


# ============================================================================
# 测试4: 门控逻辑 — 恐慌
# ============================================================================

class TestFearGating:
    """测试恐慌下的门控逻辑"""

    def test_fear_long_boost(self, gate):
        """恐慌 + 做多 → BOOST ×1.15"""
        decision = make_decision("long", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=30)
        assert result["fg_gate_action"] == FGAction.BOOST.value
        assert result["fg_gate_multiplier"] == 1.15
        assert result["quantity"] == pytest.approx(1.15)

    def test_fear_short_reduce(self, gate):
        """恐慌 + 做空 → REDUCE ×0.7"""
        decision = make_decision("short", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=35)
        assert result["fg_gate_action"] == FGAction.REDUCE.value
        assert result["fg_gate_multiplier"] == 0.7
        assert result["quantity"] == pytest.approx(0.7)


# ============================================================================
# 测试5: 门控逻辑 — 极度贪婪
# ============================================================================

class TestExtremeGreedGating:
    """测试极度贪婪下的门控逻辑"""

    def test_extreme_greed_long_reduce(self, gate):
        """极度贪婪 + 做多 → REDUCE ×0.5"""
        decision = make_decision("long", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=90)
        assert result["fg_gate_action"] == FGAction.REDUCE.value
        assert result["fg_gate_multiplier"] == 0.5
        assert result["quantity"] == pytest.approx(0.5)

    def test_extreme_greed_short_boost(self, gate):
        """极度贪婪 + 做空 → BOOST ×1.25"""
        decision = make_decision("short", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=85)
        assert result["fg_gate_action"] == FGAction.BOOST.value
        assert result["fg_gate_multiplier"] == 1.25
        assert result["quantity"] == pytest.approx(1.25)


# ============================================================================
# 测试6: 门控逻辑 — 贪婪
# ============================================================================

class TestGreedGating:
    """测试贪婪下的门控逻辑"""

    def test_greed_long_reduce(self, gate):
        """贪婪 + 做多 → REDUCE ×0.7"""
        decision = make_decision("long", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=70)
        assert result["fg_gate_action"] == FGAction.REDUCE.value
        assert result["fg_gate_multiplier"] == 0.7
        assert result["quantity"] == pytest.approx(0.7)

    def test_greed_short_boost(self, gate):
        """贪婪 + 做空 → BOOST ×1.15"""
        decision = make_decision("short", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=75)
        assert result["fg_gate_action"] == FGAction.BOOST.value
        assert result["fg_gate_multiplier"] == 1.15
        assert result["quantity"] == pytest.approx(1.15)


# ============================================================================
# 测试7: 中性 → ALLOW
# ============================================================================

class TestNeutralGating:
    """测试中性下的门控逻辑"""

    def test_neutral_long_allow(self, gate):
        """中性 + 做多 → ALLOW (不调整)"""
        decision = make_decision("long", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=50)
        assert "fg_gate_action" not in result
        assert result["quantity"] == 1.0

    def test_neutral_short_allow(self, gate):
        """中性 + 做空 → ALLOW (不调整)"""
        decision = make_decision("short", quantity=1.0)
        result = gate.apply_to_decision(decision, symbol="BTC-USDT", index_value=45)
        assert "fg_gate_action" not in result
        assert result["quantity"] == 1.0


# ============================================================================
# 测试8: disabled 开关
# ============================================================================

class TestDisabledGate:
    """测试禁用状态"""

    def test_disabled_returns_original(self, disabled_gate):
        """禁用时返回原决策"""
        decision = make_decision("long", quantity=1.0)
        result = disabled_gate.apply_to_decision(
            decision, symbol="BTC-USDT", index_value=10
        )
        assert result is decision
        assert "fg_gate_action" not in result

    def test_disabled_no_stats(self, disabled_gate):
        """禁用时不记录统计"""
        decision = make_decision("long", quantity=1.0)
        disabled_gate.apply_to_decision(decision, index_value=10)
        stats = disabled_gate.get_stats()
        assert stats["total_calls"] == 0


# ============================================================================
# 测试9: 统计追踪
# ============================================================================

class TestStats:
    """测试统计信息"""

    def test_stats_after_boost(self, gate):
        """BOOST 后统计正确"""
        gate.apply_to_decision(
            make_decision("long"), index_value=10
        )  # extreme fear → boost
        stats = gate.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_adjusted"] == 1
        assert stats["boost_count"] == 1
        assert stats["reduce_count"] == 0
        assert stats["adjust_rate"] == 1.0

    def test_stats_after_reduce(self, gate):
        """REDUCE 后统计正确"""
        gate.apply_to_decision(
            make_decision("long"), index_value=90
        )  # extreme greed → reduce
        stats = gate.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_adjusted"] == 1
        assert stats["boost_count"] == 0
        assert stats["reduce_count"] == 1

    def test_stats_after_allow(self, gate):
        """ALLOW 后统计不增加 adjusted"""
        gate.apply_to_decision(
            make_decision("long"), index_value=50
        )  # neutral → allow
        stats = gate.get_stats()
        assert stats["total_calls"] == 1
        assert stats["total_adjusted"] == 0
        assert stats["adjust_rate"] == 0.0

    def test_stats_mixed(self, gate):
        """混合调用统计正确"""
        gate.apply_to_decision(make_decision("long"), index_value=10)   # boost
        gate.apply_to_decision(make_decision("long"), index_value=90)   # reduce
        gate.apply_to_decision(make_decision("long"), index_value=50)   # allow
        stats = gate.get_stats()
        assert stats["total_calls"] == 3
        assert stats["total_adjusted"] == 2
        assert stats["boost_count"] == 1
        assert stats["reduce_count"] == 1
        assert stats["adjust_rate"] == pytest.approx(2 / 3)

    def test_reset_clears_stats(self, gate):
        """reset 清空统计"""
        gate.apply_to_decision(make_decision("long"), index_value=10)
        gate.reset()
        stats = gate.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_adjusted"] == 0
        assert stats["boost_count"] == 0
        assert stats["reduce_count"] == 0


# ============================================================================
# 测试10: 缓存加载
# ============================================================================

class TestCache:
    """测试缓存加载"""

    def test_load_from_cache_no_file(self, gate):
        """无缓存文件时返回 None"""
        # CACHE_PATH 指向的文件可能不存在
        result = gate.load_from_cache()
        # 如果文件不存在, 应返回 None
        if not os.path.exists(gate.CACHE_PATH):
            assert result is None

    def test_cache_save_and_load(self, gate, tmp_path):
        """保存后能正确加载"""
        cache_file = tmp_path / "latest.json"
        gate.CACHE_PATH = str(cache_file)
        test_data = {
            "value": 42,
            "classification": "Fear",
            "timestamp": 1234567890,
            "fetched_at": 1234567890,
        }
        gate._save_to_cache(test_data)
        loaded = gate.load_from_cache()
        assert loaded is not None
        assert loaded["value"] == 42
        assert loaded["classification"] == "Fear"


# ============================================================================
# 测试11: round_complete 回调
# ============================================================================

class TestRoundComplete:
    """测试周期完成回调"""

    def test_round_complete_no_crash(self, gate):
        """round_complete 不崩溃"""
        gate.apply_to_decision(make_decision("long"), index_value=10)
        gate.round_complete()  # 不应抛异常

    def test_round_complete_empty(self, gate):
        """空状态下 round_complete 不崩溃"""
        gate.round_complete()


# ============================================================================
# 测试12: flat 动作
# ============================================================================

class TestFlatAction:
    """测试 flat 动作"""

    def test_flat_extreme_fear_allow(self, gate):
        """flat 动作即使极度恐慌也 ALLOW"""
        decision = make_decision("flat", quantity=0.0)
        result = gate.apply_to_decision(decision, index_value=10)
        assert "fg_gate_action" not in result
        assert result["quantity"] == 0.0


# ============================================================================
# 测试13: FGAnalysisResult to_dict
# ============================================================================

class TestAnalysisResult:
    """测试 FGAnalysisResult"""

    def test_to_dict(self):
        """to_dict 返回正确格式"""
        result = FGAnalysisResult(
            index_value=25,
            level=FGLevel.FEAR,
            timestamp=1234567890.0,
            has_data=True,
        )
        d = result.to_dict()
        assert d["index_value"] == 25
        assert d["level"] == "fear"
        assert d["timestamp"] == 1234567890.0
        assert d["has_data"] is True

    def test_default_values(self):
        """默认值正确"""
        result = FGAnalysisResult()
        assert result.index_value == 50.0
        assert result.level == FGLevel.NEUTRAL
        assert result.has_data is False


# ============================================================================
# 测试14: buy/sell 动作兼容
# ============================================================================

class TestBuySellAction:
    """测试 buy/sell 动作也能正确识别"""

    def test_buy_extreme_fear_boost(self, gate):
        """buy 动作在极度恐慌时也 BOOST"""
        decision = make_decision("buy", quantity=1.0)
        result = gate.apply_to_decision(decision, index_value=10)
        assert result["fg_gate_action"] == FGAction.BOOST.value
        assert result["fg_gate_multiplier"] == 1.25

    def test_sell_extreme_greed_boost(self, gate):
        """sell 动作在极度贪婪时也 BOOST"""
        decision = make_decision("sell", quantity=1.0)
        result = gate.apply_to_decision(decision, index_value=90)
        assert result["fg_gate_action"] == FGAction.BOOST.value
        assert result["fg_gate_multiplier"] == 1.25


# ============================================================================
# 测试15: 合成数据生成
# ============================================================================

class TestSyntheticIndex:
    """测试合成指数生成"""

    def test_synthetic_in_range(self, gate):
        """合成指数在 0-100 范围内"""
        for _ in range(20):
            idx = gate._generate_synthetic_index()
            assert 0.0 <= idx <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
