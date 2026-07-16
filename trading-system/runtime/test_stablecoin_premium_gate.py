#!/usr/bin/env python3
"""StablecoinPremiumGate 单元测试"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stablecoin_premium_gate import (
    SPPremiumLevel,
    SPAnalysisResult,
    StablecoinPremiumGate,
)


class TestSPAnalysisResult(unittest.TestCase):
    """测试 SPAnalysisResult"""

    def test_default_values(self):
        r = SPAnalysisResult()
        self.assertEqual(r.usdt_price, 1.0)
        self.assertEqual(r.premium_pct, 0.0)
        self.assertEqual(r.level, SPPremiumLevel.NORMAL)
        self.assertFalse(r.has_data)

    def test_to_dict(self):
        r = SPAnalysisResult(usdt_price=1.005, premium_pct=0.005, has_data=True)
        d = r.to_dict()
        self.assertEqual(d["usdt_price"], 1.005)
        self.assertTrue(d["has_data"])


class TestStablecoinPremiumGateInit(unittest.TestCase):
    """测试初始化"""

    def test_default_init(self):
        gate = StablecoinPremiumGate()
        self.assertTrue(gate.enabled)
        self.assertEqual(gate._total_calls, 0)

    def test_disabled_init(self):
        gate = StablecoinPremiumGate(enabled=False)
        self.assertFalse(gate.enabled)

    def test_thresholds(self):
        gate = StablecoinPremiumGate()
        self.assertEqual(gate.PREMIUM_HIGH, 0.005)
        self.assertEqual(gate.PREMIUM_EXTREME, 0.01)
        self.assertEqual(gate.DISCOUNT_HIGH, -0.005)
        self.assertEqual(gate.DISCOUNT_EXTREME, -0.01)
        self.assertEqual(gate.BOOST_MULTIPLIER, 1.15)
        self.assertEqual(gate.REDUCE_MULTIPLIER, 0.7)
        self.assertEqual(gate.EXTREME_REDUCE_MULTIPLIER, 0.5)


class TestAnalyzePremium(unittest.TestCase):
    """测试溢价分析"""

    def test_zero_price(self):
        gate = StablecoinPremiumGate()
        result = gate.analyze_premium(0)
        self.assertFalse(result.has_data)

    def test_normal_premium(self):
        gate = StablecoinPremiumGate()
        result = gate.analyze_premium(1.0001)
        self.assertTrue(result.has_data)
        self.assertEqual(result.level, SPPremiumLevel.NORMAL)

    def test_high_premium(self):
        gate = StablecoinPremiumGate()
        result = gate.analyze_premium(1.006)  # 0.6%溢价
        self.assertEqual(result.level, SPPremiumLevel.HIGH_PREMIUM)
        self.assertGreater(result.premium_pct, gate.PREMIUM_HIGH)

    def test_extreme_premium(self):
        gate = StablecoinPremiumGate()
        result = gate.analyze_premium(1.015)  # 1.5%溢价
        self.assertEqual(result.level, SPPremiumLevel.EXTREME_PREMIUM)

    def test_high_discount(self):
        gate = StablecoinPremiumGate()
        result = gate.analyze_premium(0.994)  # 0.6%折价
        self.assertEqual(result.level, SPPremiumLevel.HIGH_DISCOUNT)

    def test_extreme_discount(self):
        gate = StablecoinPremiumGate()
        result = gate.analyze_premium(0.985)  # 1.5%折价
        self.assertEqual(result.level, SPPremiumLevel.EXTREME_DISCOUNT)


class TestApplyToDecision(unittest.TestCase):
    """测试决策门控"""

    def test_disabled_gate(self):
        gate = StablecoinPremiumGate(enabled=False)
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.01)
        self.assertEqual(result["quantity"], 1.0)

    def test_extreme_premium_short_boost(self):
        """规则1: 极端溢价 + 做空 → BOOST"""
        gate = StablecoinPremiumGate()
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.015)
        self.assertEqual(result["sp_gate_action"], "boost")
        self.assertAlmostEqual(result["sp_gate_multiplier"], 1.15, places=2)

    def test_extreme_premium_long_reduce(self):
        """规则1: 极端溢价 + 做多 → EXTREME_REDUCE"""
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.015)
        self.assertEqual(result["sp_gate_action"], "reduce")
        self.assertAlmostEqual(result["sp_gate_multiplier"], 0.5, places=2)

    def test_high_premium_short_boost(self):
        """规则2: 高溢价 + 做空 → BOOST"""
        gate = StablecoinPremiumGate()
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.006)
        self.assertEqual(result["sp_gate_action"], "boost")

    def test_high_premium_long_reduce(self):
        """规则2: 高溢价 + 做多 → REDUCE"""
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.006)
        self.assertEqual(result["sp_gate_action"], "reduce")
        self.assertAlmostEqual(result["sp_gate_multiplier"], 0.7, places=2)

    def test_extreme_discount_long_boost(self):
        """规则3: 极端折价 + 做多 → BOOST"""
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=0.985)
        self.assertEqual(result["sp_gate_action"], "boost")
        self.assertAlmostEqual(result["sp_gate_multiplier"], 1.15, places=2)

    def test_extreme_discount_short_reduce(self):
        """规则3: 极端折价 + 做空 → EXTREME_REDUCE"""
        gate = StablecoinPremiumGate()
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=0.985)
        self.assertEqual(result["sp_gate_action"], "reduce")
        self.assertAlmostEqual(result["sp_gate_multiplier"], 0.5, places=2)

    def test_high_discount_long_boost(self):
        """规则4: 高折价 + 做多 → BOOST"""
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=0.994)
        self.assertEqual(result["sp_gate_action"], "boost")

    def test_high_discount_short_reduce(self):
        """规则4: 高折价 + 做空 → REDUCE"""
        gate = StablecoinPremiumGate()
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=0.994)
        self.assertEqual(result["sp_gate_action"], "reduce")

    def test_normal_allow(self):
        """正常 → ALLOW"""
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.0001)
        self.assertNotIn("sp_gate_action", result)

    def test_synthetic_price_used(self):
        """未传usdt_price时使用合成价格"""
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision)
        self.assertIsNotNone(result)
        self.assertGreater(gate._total_calls, 0)


class TestStatsAndReset(unittest.TestCase):
    """测试统计和重置"""

    def test_stats_initial(self):
        gate = StablecoinPremiumGate()
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["total_adjusted"], 0)

    def test_stats_after_calls(self):
        gate = StablecoinPremiumGate()
        decision = {"action": "sell", "quantity": 1.0}
        gate.apply_to_decision(decision, usdt_price=1.015)
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["total_adjusted"], 1)
        self.assertEqual(stats["boost_count"], 1)

    def test_reset(self):
        gate = StablecoinPremiumGate()
        gate._total_calls = 10
        gate.reset()
        self.assertEqual(gate._total_calls, 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_zero_quantity(self):
        gate = StablecoinPremiumGate()
        decision = {"action": "buy", "quantity": 0.0}
        result = gate.apply_to_decision(decision, usdt_price=1.015)
        self.assertEqual(result["quantity"], 0.0)

    def test_long_action_variants(self):
        gate = StablecoinPremiumGate()
        for action in ["buy", "long", "BUY", "LONG"]:
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, usdt_price=1.015)
            self.assertEqual(
                result["sp_gate_action"], "reduce",
                f"Failed for action={action}",
            )

    def test_short_action_variants(self):
        gate = StablecoinPremiumGate()
        for action in ["sell", "short", "SELL", "SHORT"]:
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, usdt_price=1.015)
            self.assertEqual(
                result["sp_gate_action"], "boost",
                f"Failed for action={action}",
            )

    def test_neutral_action_no_trigger(self):
        """中性action不触发"""
        gate = StablecoinPremiumGate()
        decision = {"action": "neutral", "quantity": 1.0}
        result = gate.apply_to_decision(decision, usdt_price=1.015)
        self.assertNotIn("sp_gate_action", result)


class TestMultiRoundStability(unittest.TestCase):
    """多轮稳定性"""

    def test_100_rounds_no_crash(self):
        gate = StablecoinPremiumGate()
        for i in range(100):
            decision = {"action": "buy" if i % 2 == 0 else "sell", "quantity": 1.0}
            # 交替使用不同价格
            price = 1.015 if i % 4 == 0 else (0.985 if i % 4 == 1 else 1.0001)
            result = gate.apply_to_decision(decision, usdt_price=price)
            self.assertIsNotNone(result)
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 100)
        self.assertGreater(stats["total_adjusted"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
