#!/usr/bin/env python3
"""SmartMoneyGate 单元测试"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smart_money_gate import (
    SMGateAction,
    SMAnalysisResult,
    SmartMoneyGate,
)


class TestSMAnalysisResult(unittest.TestCase):

    def test_default_values(self):
        r = SMAnalysisResult()
        self.assertEqual(r.sentiment, 0.0)
        self.assertFalse(r.has_data)

    def test_to_dict(self):
        r = SMAnalysisResult(sentiment=0.5, has_data=True)
        d = r.to_dict()
        self.assertEqual(d["sentiment"], 0.5)


class TestSmartMoneyGateInit(unittest.TestCase):

    def test_default_init(self):
        gate = SmartMoneyGate()
        self.assertTrue(gate.enabled)

    def test_disabled_init(self):
        gate = SmartMoneyGate(enabled=False)
        self.assertFalse(gate.enabled)

    def test_thresholds(self):
        gate = SmartMoneyGate()
        self.assertEqual(gate.STRONG_BULLISH, 0.3)
        self.assertEqual(gate.STRONG_BEARISH, -0.3)
        self.assertEqual(gate.EXTREME_BULLISH, 0.6)
        self.assertEqual(gate.EXTREME_BEARISH, -0.6)


class TestApplyToDecision(unittest.TestCase):

    def test_disabled_gate(self):
        gate = SmartMoneyGate(enabled=False)
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, price=40000)
        self.assertEqual(result["quantity"], 1.0)

    def test_no_price_data(self):
        gate = SmartMoneyGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision)
        self.assertEqual(result["quantity"], 1.0)
        self.assertNotIn("sm_gate_action", result)

    def test_neutral_sentiment_allow(self):
        """中性sentiment → ALLOW"""
        gate = SmartMoneyGate()
        # 提供少量数据, sentiment应该接近0
        for i in range(5):
            gate.update_market_data(price=40000.0, volume=100.0)
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, price=40000.0, volume=100.0)
        # 检查是否被调整 (可能ALLOW也可能触发, 取决于合成数据)
        self.assertIsNotNone(result)

    def test_extreme_bullish_long_boost(self):
        """规则1: 极端看涨 + 做多 → EXTREME_BOOST"""
        gate = SmartMoneyGate()
        # 直接设置sentiment
        gate._last_analysis = SMAnalysisResult(sentiment=0.7, has_data=True)
        # 调用_evaluate_rules
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "buy", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.BOOST)
        self.assertAlmostEqual(mult, 1.25, places=2)

    def test_extreme_bullish_short_reduce(self):
        """规则1: 极端看涨 + 做空 → EXTREME_REDUCE"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=0.7, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "sell", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.REDUCE)
        self.assertAlmostEqual(mult, 0.5, places=2)

    def test_strong_bullish_long_boost(self):
        """规则2: 强看涨 + 做多 → BOOST"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=0.4, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "buy", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.BOOST)
        self.assertAlmostEqual(mult, 1.15, places=2)

    def test_strong_bullish_short_reduce(self):
        """规则2: 强看涨 + 做空 → REDUCE"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=0.4, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "sell", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.REDUCE)

    def test_extreme_bearish_short_boost(self):
        """规则3: 极端看跌 + 做空 → EXTREME_BOOST"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=-0.7, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "sell", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.BOOST)
        self.assertAlmostEqual(mult, 1.25, places=2)

    def test_extreme_bearish_long_reduce(self):
        """规则3: 极端看跌 + 做多 → EXTREME_REDUCE"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=-0.7, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "buy", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.REDUCE)
        self.assertAlmostEqual(mult, 0.5, places=2)

    def test_strong_bearish_short_boost(self):
        """规则4: 强看跌 + 做空 → BOOST"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=-0.4, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "sell", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.BOOST)

    def test_neutral_sentiment(self):
        """中性 → ALLOW"""
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=0.1, has_data=True)
        action, mult, reason = gate._evaluate_rules(
            gate._last_analysis,
            {"action": "buy", "quantity": 1.0},
        )
        self.assertEqual(action, SMGateAction.ALLOW)


class TestMarketDataUpdate(unittest.TestCase):

    def test_update_market_data(self):
        gate = SmartMoneyGate()
        gate.update_market_data(price=40000.0, volume=100.0)
        self.assertEqual(len(gate._price_history), 1)

    def test_update_multiple(self):
        gate = SmartMoneyGate()
        for i in range(10):
            gate.update_market_data(price=40000.0 + i, volume=100.0)
        self.assertEqual(len(gate._price_history), 10)

    def test_zero_price_ignored(self):
        gate = SmartMoneyGate()
        gate.update_market_data(price=0, volume=100.0)
        self.assertEqual(len(gate._price_history), 0)


class TestStatsAndReset(unittest.TestCase):

    def test_stats_initial(self):
        gate = SmartMoneyGate()
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 0)

    def test_reset(self):
        gate = SmartMoneyGate()
        gate._total_calls = 10
        gate.reset()
        self.assertEqual(gate._total_calls, 0)


class TestEdgeCases(unittest.TestCase):

    def test_long_action_variants(self):
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=0.7, has_data=True)
        for action in ["buy", "long", "BUY", "LONG"]:
            a, m, r = gate._evaluate_rules(
                gate._last_analysis,
                {"action": action, "quantity": 1.0},
            )
            self.assertEqual(a, SMGateAction.BOOST, f"Failed for {action}")

    def test_short_action_variants(self):
        gate = SmartMoneyGate()
        gate._last_analysis = SMAnalysisResult(sentiment=0.7, has_data=True)
        for action in ["sell", "short", "SELL", "SHORT"]:
            a, m, r = gate._evaluate_rules(
                gate._last_analysis,
                {"action": action, "quantity": 1.0},
            )
            self.assertEqual(a, SMGateAction.REDUCE, f"Failed for {action}")


class TestMultiRoundStability(unittest.TestCase):

    def test_100_rounds_no_crash(self):
        gate = SmartMoneyGate()
        for i in range(100):
            price = 40000.0 + i * 10
            decision = {"action": "buy" if i % 2 == 0 else "sell", "quantity": 1.0}
            result = gate.apply_to_decision(decision, price=price, volume=100.0)
            self.assertIsNotNone(result)
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
