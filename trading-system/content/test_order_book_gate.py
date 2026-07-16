#!/usr/bin/env python3
"""OrderBookGate 单元测试

测试覆盖:
  - OBAnalysisResult 数据结构
  - OrderBookGate 初始化
  - analyze_order_book 分析逻辑
  - 大单墙检测
  - apply_to_decision 门控规则 (10条)
  - 边界情况
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from order_book_gate import (
    OBGateAction,
    OBAnalysisResult,
    OBGateResult,
    OrderBookGate,
)


class TestOBAnalysisResult(unittest.TestCase):
    """测试 OBAnalysisResult 数据结构"""

    def test_default_values(self):
        result = OBAnalysisResult()
        self.assertEqual(result.mid_price, 0.0)
        self.assertEqual(result.spread, 0.0)
        self.assertEqual(result.spread_pct, 0.0)
        self.assertEqual(result.bid_depth, 0.0)
        self.assertEqual(result.ask_depth, 0.0)
        self.assertEqual(result.imbalance, 0.0)
        self.assertIsNone(result.bid_wall)
        self.assertIsNone(result.ask_wall)
        self.assertFalse(result.has_data)

    def test_to_dict(self):
        result = OBAnalysisResult(
            mid_price=40000.0,
            spread=2.0,
            spread_pct=0.00005,
            has_data=True,
        )
        d = result.to_dict()
        self.assertEqual(d["mid_price"], 40000.0)
        self.assertEqual(d["spread"], 2.0)
        self.assertTrue(d["has_data"])


class TestOBGateResult(unittest.TestCase):
    """测试 OBGateResult 数据结构"""

    def test_default_values(self):
        result = OBGateResult()
        self.assertEqual(result.action, OBGateAction.ALLOW)
        self.assertEqual(result.multiplier, 1.0)
        self.assertEqual(result.reason, "")
        self.assertIsNone(result.analysis)

    def test_to_dict(self):
        result = OBGateResult(
            action=OBGateAction.BOOST,
            multiplier=1.15,
            reason="test",
        )
        d = result.to_dict()
        self.assertEqual(d["action"], "boost")
        self.assertEqual(d["multiplier"], 1.15)
        self.assertEqual(d["reason"], "test")


class TestOrderBookGateInit(unittest.TestCase):
    """测试 OrderBookGate 初始化"""

    def test_default_init(self):
        gate = OrderBookGate()
        self.assertTrue(gate.enabled)
        self.assertEqual(gate._total_calls, 0)
        self.assertEqual(gate._total_adjusted, 0)

    def test_disabled_init(self):
        gate = OrderBookGate(enabled=False)
        self.assertFalse(gate.enabled)

    def test_thresholds(self):
        gate = OrderBookGate()
        self.assertEqual(gate.IMBALANCE_STRONG, 0.3)
        self.assertEqual(gate.IMBALANCE_EXTREME, 0.5)
        self.assertEqual(gate.WALL_RATIO_THRESHOLD, 3.0)
        self.assertEqual(gate.SPREAD_PCT_HIGH, 0.002)
        self.assertEqual(gate.SPREAD_PCT_EXTREME, 0.005)
        self.assertEqual(gate.BOOST_MULTIPLIER, 1.15)
        self.assertEqual(gate.REDUCE_MULTIPLIER, 0.7)
        self.assertEqual(gate.EXTREME_REDUCE_MULTIPLIER, 0.5)


class TestAnalyzeOrderBook(unittest.TestCase):
    """测试订单簿分析"""

    def test_empty_bids_asks(self):
        gate = OrderBookGate()
        result = gate.analyze_order_book([], [])
        self.assertFalse(result.has_data)

    def test_basic_analysis(self):
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 1.5), (39996.0, 2.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.5), (40004.0, 2.0)]
        result = gate.analyze_order_book(bids, asks)
        self.assertTrue(result.has_data)
        self.assertAlmostEqual(result.mid_price, 40000.0, places=1)
        self.assertGreater(result.spread, 0)
        self.assertGreater(result.spread_pct, 0)
        self.assertGreater(result.bid_depth, 0)
        self.assertGreater(result.ask_depth, 0)

    def test_imbalance_calculation(self):
        gate = OrderBookGate()
        # 买盘多于卖盘
        bids = [(39998.0, 3.0), (39997.0, 3.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0)]
        result = gate.analyze_order_book(bids, asks)
        self.assertGreater(result.imbalance, 0)  # 买盘多 → 正不平衡

        # 卖盘多于买盘
        bids2 = [(39998.0, 1.0), (39997.0, 1.0)]
        asks2 = [(40002.0, 3.0), (40003.0, 3.0)]
        result2 = gate.analyze_order_book(bids2, asks2)
        self.assertLess(result2.imbalance, 0)  # 卖盘多 → 负不平衡

    def test_spread_pct(self):
        gate = OrderBookGate()
        bids = [(39990.0, 1.0)]
        asks = [(40010.0, 1.0)]
        result = gate.analyze_order_book(bids, asks)
        # mid = 40000, spread = 20, spread_pct = 20/40000 = 0.0005
        self.assertAlmostEqual(result.spread_pct, 0.0005, places=6)


class TestWallDetection(unittest.TestCase):
    """测试大单墙检测"""

    def test_no_wall(self):
        gate = OrderBookGate()
        levels = [(100.0, 1.0), (99.0, 1.1), (98.0, 1.2)]
        wall, ratio = gate._detect_wall(levels)
        self.assertIsNone(wall)
        self.assertEqual(ratio, 0.0)

    def test_wall_detected(self):
        gate = OrderBookGate()
        # 最大挂单量是其他的5倍
        levels = [(100.0, 1.0), (99.0, 5.0), (98.0, 1.0)]
        wall, ratio = gate._detect_wall(levels)
        self.assertIsNotNone(wall)
        self.assertGreaterEqual(ratio, gate.WALL_RATIO_THRESHOLD)

    def test_single_level(self):
        gate = OrderBookGate()
        levels = [(100.0, 1.0)]
        wall, ratio = gate._detect_wall(levels)
        self.assertIsNone(wall)
        self.assertEqual(ratio, 0.0)

    def test_bid_wall_in_analysis(self):
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        result = gate.analyze_order_book(bids, asks)
        self.assertIsNotNone(result.bid_wall)


class TestApplyToDecision(unittest.TestCase):
    """测试决策门控规则"""

    def test_disabled_gate(self):
        gate = OrderBookGate(enabled=False)
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=[(100, 1)], asks=[(101, 1)])
        self.assertEqual(result["quantity"], 1.0)

    def test_no_order_book_data(self):
        gate = OrderBookGate()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision)
        self.assertEqual(result["quantity"], 1.0)
        self.assertNotIn("ob_gate_action", result)

    def test_rule_extreme_spread(self):
        """规则1: 极端价差 → EXTREME_REDUCE"""
        gate = OrderBookGate()
        # mid=10000, spread=60, spread_pct=0.006 > 0.005 (EXTREME)
        bids = [(9970.0, 1.0)]
        asks = [(10030.0, 1.0)]
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "reduce")
        self.assertAlmostEqual(result["ob_gate_multiplier"], 0.5, places=2)

    def test_rule_high_spread(self):
        """规则2: 高价差 → REDUCE"""
        gate = OrderBookGate()
        # spread_pct = 50/10000 = 0.005, 不超过extreme, 但超过high(0.002)
        # 实际上0.005 = extreme阈值, 用0.003
        bids = [(9985.0, 1.0)]
        asks = [(10015.0, 1.0)]
        # spread = 30, mid = 10000, spread_pct = 0.003
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "reduce")
        self.assertAlmostEqual(result["ob_gate_multiplier"], 0.7, places=2)

    def test_rule_bid_wall_long_boost(self):
        """规则3: 买盘墙 + 做多 → BOOST"""
        gate = OrderBookGate()
        # 构造买盘墙
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        # spread_pct = 4/40000 = 0.0001, 不会触发spread规则
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "boost")
        self.assertAlmostEqual(result["ob_gate_multiplier"], 1.15, places=2)

    def test_rule_bid_wall_short_reduce(self):
        """规则4: 买盘墙 + 做空 → REDUCE"""
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "reduce")

    def test_rule_ask_wall_short_boost(self):
        """规则5: 卖盘墙 + 做空 → BOOST"""
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 1.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 10.0), (40004.0, 1.0), (40005.0, 1.0)]
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "boost")

    def test_rule_ask_wall_long_reduce(self):
        """规则6: 卖盘墙 + 做多 → REDUCE"""
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 1.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 10.0), (40004.0, 1.0), (40005.0, 1.0)]
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "reduce")

    def test_rule_strong_bid_imbalance_long_boost(self):
        """规则7: 强买盘不平衡 + 做多 → BOOST"""
        gate = OrderBookGate()
        # bid_depth=10, ask_depth=2, imbalance = 8/12 = 0.67 > 0.3
        bids = [(39998.0, 2.5), (39997.0, 2.5), (39996.0, 2.5), (39995.0, 2.5)]
        asks = [(40002.0, 0.5), (40003.0, 0.5), (40004.0, 0.5), (40005.0, 0.5)]
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "boost")

    def test_rule_strong_ask_imbalance_short_boost(self):
        """规则9: 强卖盘不平衡 + 做空 → BOOST"""
        gate = OrderBookGate()
        bids = [(39998.0, 0.5), (39997.0, 0.5), (39996.0, 0.5), (39995.0, 0.5)]
        asks = [(40002.0, 2.5), (40003.0, 2.5), (40004.0, 2.5), (40005.0, 2.5)]
        decision = {"action": "sell", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["ob_gate_action"], "boost")

    def test_no_trigger_allow(self):
        """无触发 → ALLOW"""
        gate = OrderBookGate()
        # 均衡的订单簿，无大单墙，价差正常
        bids = [(39998.0, 1.0), (39997.0, 1.0), (39996.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0)]
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertNotIn("ob_gate_action", result)
        self.assertEqual(result["quantity"], 1.0)


class TestOrderBookObjectInput(unittest.TestCase):
    """测试通过order_book对象传入"""

    def test_object_with_bids_asks_attributes(self):
        gate = OrderBookGate()
        class MockOB:
            def __init__(self):
                self.bids = [(39998.0, 1.0), (39997.0, 1.0)]
                self.asks = [(40002.0, 1.0), (40003.0, 1.0)]
        ob = MockOB()
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, order_book=ob)
        # 均衡订单簿 → ALLOW
        self.assertNotIn("ob_gate_action", result)

    def test_dict_order_book(self):
        gate = OrderBookGate()
        ob = {
            "bids": [(39998.0, 1.0), (39997.0, 1.0)],
            "asks": [(40002.0, 1.0), (40003.0, 1.0)],
        }
        decision = {"action": "buy", "quantity": 1.0}
        result = gate.apply_to_decision(decision, order_book=ob)
        self.assertIsNotNone(result)


class TestSyntheticOrderBook(unittest.TestCase):
    """测试模拟订单簿生成"""

    def test_generate_basic(self):
        gate = OrderBookGate()
        bids, asks = gate.generate_synthetic_order_book(
            mid_price=40000.0,
            spread_pct=0.0005,
            depth_levels=5,
        )
        self.assertEqual(len(bids), 5)
        self.assertEqual(len(asks), 5)
        self.assertGreater(bids[0][0], bids[-1][0])  # bid降序
        self.assertLess(asks[0][0], asks[-1][0])   # ask升序

    def test_generate_with_imbalance(self):
        gate = OrderBookGate()
        bids, asks = gate.generate_synthetic_order_book(
            mid_price=40000.0,
            imbalance=0.5,  # 买盘多
        )
        result = gate.analyze_order_book(bids, asks)
        self.assertGreater(result.imbalance, 0)

    def test_generate_zero_price(self):
        gate = OrderBookGate()
        bids, asks = gate.generate_synthetic_order_book(mid_price=0)
        self.assertEqual(len(bids), 0)
        self.assertEqual(len(asks), 0)


class TestStatsAndReset(unittest.TestCase):
    """测试统计和重置"""

    def test_stats_initial(self):
        gate = OrderBookGate()
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["total_adjusted"], 0)
        self.assertEqual(stats["boost_count"], 0)
        self.assertEqual(stats["reduce_count"], 0)

    def test_stats_after_calls(self):
        gate = OrderBookGate()
        # 触发BOOST
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        decision = {"action": "buy", "quantity": 1.0}
        gate.apply_to_decision(decision, bids=bids, asks=asks)
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["total_adjusted"], 1)
        self.assertEqual(stats["boost_count"], 1)

    def test_reset(self):
        gate = OrderBookGate()
        gate._total_calls = 10
        gate._total_adjusted = 5
        gate.reset()
        self.assertEqual(gate._total_calls, 0)
        self.assertEqual(gate._total_adjusted, 0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_negative_quantity_after_reduce(self):
        gate = OrderBookGate()
        bids = [(9995.0, 1.0)]
        asks = [(10005.0, 1.0)]
        decision = {"action": "buy", "quantity": 0.1}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        # 0.1 * 0.5 = 0.05, 不会负
        self.assertGreater(result["quantity"], 0)

    def test_zero_quantity(self):
        gate = OrderBookGate()
        bids = [(9995.0, 1.0)]
        asks = [(10005.0, 1.0)]
        decision = {"action": "buy", "quantity": 0.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        self.assertEqual(result["quantity"], 0.0)

    def test_long_action_variants(self):
        """测试不同的做多action表述"""
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        for action in ["buy", "long", "BUY", "LONG", "Buy"]:
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, bids=bids, asks=asks)
            self.assertEqual(
                result["ob_gate_action"], "boost",
                f"Failed for action={action}",
            )

    def test_short_action_variants(self):
        """测试不同的做空action表述"""
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        for action in ["sell", "short", "SELL", "SHORT", "Sell"]:
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, bids=bids, asks=asks)
            self.assertEqual(
                result["ob_gate_action"], "reduce",
                f"Failed for action={action}",
            )

    def test_neutral_action_no_trigger(self):
        """中性action不触发墙规则"""
        gate = OrderBookGate()
        bids = [(39998.0, 1.0), (39997.0, 10.0), (39996.0, 1.0), (39995.0, 1.0)]
        asks = [(40002.0, 1.0), (40003.0, 1.0), (40004.0, 1.0), (40005.0, 1.0)]
        decision = {"action": "neutral", "quantity": 1.0}
        result = gate.apply_to_decision(decision, bids=bids, asks=asks)
        # neutral不匹配long/short, 所以不会触发墙规则
        self.assertNotIn("ob_gate_action", result)


class TestMultiRoundStability(unittest.TestCase):
    """多轮调用稳定性"""

    def test_100_rounds_no_crash(self):
        gate = OrderBookGate()
        for i in range(100):
            bids, asks = gate.generate_synthetic_order_book(
                mid_price=40000.0 + i,
                spread_pct=0.0005,
                imbalance=0.1 if i % 2 == 0 else -0.1,
            )
            decision = {"action": "buy" if i % 2 == 0 else "sell", "quantity": 1.0}
            result = gate.apply_to_decision(decision, bids=bids, asks=asks)
            self.assertIsNotNone(result)
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
