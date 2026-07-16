#!/usr/bin/env python3
"""InstitutionRetailGate 单元测试"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from institution_retail_gate import (
    IRLevel,
    IRAction,
    IRAnalysisResult,
    InstitutionRetailGate,
)


def make_analysis(ratio: float, has_data: bool = True) -> IRAnalysisResult:
    """构造指定 institution_ratio 的 IRAnalysisResult (用于测试)"""
    buy = ratio * 100.0
    sell = (1.0 - ratio) * 100.0
    return IRAnalysisResult(
        taker_buy_volume=buy,
        taker_sell_volume=sell,
        total_volume=buy + sell,
        institution_ratio=ratio,
        buy_sell_ratio=buy / sell if sell > 0 else float("inf"),
        level=InstitutionRetailGate._classify_level(ratio, buy + sell),
        has_data=has_data,
    )


class TestIRAnalysisResult(unittest.TestCase):
    """测试 IRAnalysisResult"""

    def test_default_values(self):
        r = IRAnalysisResult()
        self.assertEqual(r.taker_buy_volume, 0.0)
        self.assertEqual(r.taker_sell_volume, 0.0)
        self.assertEqual(r.total_volume, 0.0)
        self.assertEqual(r.institution_ratio, 0.5)
        self.assertEqual(r.buy_sell_ratio, 1.0)
        self.assertEqual(r.level, IRLevel.BALANCED)
        self.assertFalse(r.has_data)

    def test_to_dict(self):
        r = IRAnalysisResult(
            taker_buy_volume=70.0,
            taker_sell_volume=30.0,
            total_volume=100.0,
            institution_ratio=0.7,
            buy_sell_ratio=2.333,
            level=IRLevel.INSTITUTION_BUY,
            has_data=True,
        )
        d = r.to_dict()
        self.assertEqual(d["taker_buy_volume"], 70.0)
        self.assertEqual(d["total_volume"], 100.0)
        self.assertAlmostEqual(d["institution_ratio"], 0.7, places=2)
        self.assertEqual(d["level"], "institution_buy")
        self.assertTrue(d["has_data"])


class TestInstitutionRetailGateInit(unittest.TestCase):
    """测试初始化"""

    def test_default_init(self):
        gate = InstitutionRetailGate()
        self.assertTrue(gate.enabled)
        self.assertEqual(gate._total_calls, 0)
        self.assertEqual(gate._total_adjusted, 0)

    def test_disabled_init(self):
        gate = InstitutionRetailGate(enabled=False)
        self.assertFalse(gate.enabled)

    def test_thresholds(self):
        gate = InstitutionRetailGate()
        self.assertEqual(gate.INSTITUTION_BUY_THRESHOLD, 0.60)
        self.assertEqual(gate.INSTITUTION_SELL_THRESHOLD, 0.40)
        self.assertEqual(gate.STRONG_BUY_THRESHOLD, 0.70)
        self.assertEqual(gate.STRONG_SELL_THRESHOLD, 0.30)
        self.assertEqual(gate.BOOST_MULTIPLIER, 1.15)
        self.assertEqual(gate.REDUCE_MULTIPLIER, 0.7)
        self.assertEqual(gate.STRONG_BOOST_MULTIPLIER, 1.25)
        self.assertEqual(gate.STRONG_REDUCE_MULTIPLIER, 0.5)


class TestSymbolNormalization(unittest.TestCase):
    """测试 Symbol 标准化"""

    def test_normalize_symbol_variants(self):
        cases = [
            ("BTC", "BTCUSDT"),
            ("BTC-USDT", "BTCUSDT"),
            ("BTC/USDT", "BTCUSDT"),
            ("BTC_USDT", "BTCUSDT"),
            ("btc-usdt", "BTCUSDT"),
            ("BTCUSDT", "BTCUSDT"),
            ("", "BTCUSDT"),
            ("eth", "ETHUSDT"),
        ]
        for inp, expected in cases:
            self.assertEqual(
                InstitutionRetailGate._normalize_symbol(inp), expected,
                f"Failed for input={inp!r}",
            )

    def test_cache_symbol_variants(self):
        cases = [
            ("BTC", "BTC"),
            ("BTC-USDT", "BTC"),
            ("BTC/USDT", "BTC"),
            ("BTCUSDT", "BTC"),
            ("ETH-USDT", "ETH"),
        ]
        for inp, expected in cases:
            self.assertEqual(
                InstitutionRetailGate._cache_symbol(inp), expected,
                f"Failed for input={inp!r}",
            )


class TestParseVolumeData(unittest.TestCase):
    """测试 volume 数据解析"""

    def test_valid_data(self):
        gate = InstitutionRetailGate()
        data = {
            "buySellRatio": "1.8571",
            "buyVol": "65.00",
            "sellVol": "35.00",
            "timestamp": "1700000000000",
        }
        result = gate._parse_volume_data(data)
        self.assertTrue(result.has_data)
        self.assertEqual(result.taker_buy_volume, 65.0)
        self.assertEqual(result.taker_sell_volume, 35.0)
        self.assertEqual(result.total_volume, 100.0)
        self.assertAlmostEqual(result.institution_ratio, 0.65, places=2)
        self.assertAlmostEqual(result.buy_sell_ratio, 65.0 / 35.0, places=2)
        self.assertEqual(result.level, IRLevel.INSTITUTION_BUY)

    def test_zero_volume(self):
        gate = InstitutionRetailGate()
        data = {"buySellRatio": "0", "buyVol": "0", "sellVol": "0", "timestamp": "0"}
        result = gate._parse_volume_data(data)
        self.assertFalse(result.has_data)
        self.assertEqual(result.level, IRLevel.RETAIL_DOMINANT)

    def test_negative_volume(self):
        gate = InstitutionRetailGate()
        data = {"buyVol": "-10", "sellVol": "20", "timestamp": "0"}
        result = gate._parse_volume_data(data)
        self.assertFalse(result.has_data)

    def test_invalid_data(self):
        gate = InstitutionRetailGate()
        data = {"buyVol": "abc", "sellVol": "xyz", "timestamp": "0"}
        result = gate._parse_volume_data(data)
        self.assertFalse(result.has_data)

    def test_balanced_ratio(self):
        gate = InstitutionRetailGate()
        data = {"buyVol": "50", "sellVol": "50", "timestamp": "0"}
        result = gate._parse_volume_data(data)
        self.assertTrue(result.has_data)
        self.assertAlmostEqual(result.institution_ratio, 0.5, places=2)
        self.assertEqual(result.level, IRLevel.BALANCED)


class TestClassifyLevel(unittest.TestCase):
    """测试等级分类"""

    def test_institution_buy(self):
        level = InstitutionRetailGate._classify_level(0.65, 100.0)
        self.assertEqual(level, IRLevel.INSTITUTION_BUY)

    def test_institution_sell(self):
        level = InstitutionRetailGate._classify_level(0.35, 100.0)
        self.assertEqual(level, IRLevel.INSTITUTION_SELL)

    def test_balanced(self):
        level = InstitutionRetailGate._classify_level(0.50, 100.0)
        self.assertEqual(level, IRLevel.BALANCED)

    def test_retail_dominant_low_volume(self):
        level = InstitutionRetailGate._classify_level(0.50, 0.5)
        self.assertEqual(level, IRLevel.RETAIL_DOMINANT)


class TestEvaluateRules(unittest.TestCase):
    """测试门控规则评估 (核心逻辑)"""

    def test_strong_inst_buy_long_boost(self):
        """规则1: 强机构买入 (ratio>0.7) + 做多 → BOOST ×1.25"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.75)
        decision = {"action": "long", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.BOOST)
        self.assertAlmostEqual(mult, 1.25, places=2)

    def test_strong_inst_buy_short_reduce(self):
        """规则1: 强机构买入 (ratio>0.7) + 做空 → REDUCE ×0.5"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.75)
        decision = {"action": "short", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.REDUCE)
        self.assertAlmostEqual(mult, 0.5, places=2)

    def test_inst_buy_long_boost(self):
        """规则2: 机构买入 (ratio>0.6) + 做多 → BOOST ×1.15"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.65)
        decision = {"action": "long", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.BOOST)
        self.assertAlmostEqual(mult, 1.15, places=2)

    def test_inst_buy_short_reduce(self):
        """规则2: 机构买入 (ratio>0.6) + 做空 → REDUCE ×0.7"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.65)
        decision = {"action": "short", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.REDUCE)
        self.assertAlmostEqual(mult, 0.7, places=2)

    def test_strong_inst_sell_long_reduce(self):
        """规则3: 强机构卖出 (ratio<0.3) + 做多 → REDUCE ×0.5"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.25)
        decision = {"action": "long", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.REDUCE)
        self.assertAlmostEqual(mult, 0.5, places=2)

    def test_strong_inst_sell_short_boost(self):
        """规则3: 强机构卖出 (ratio<0.3) + 做空 → BOOST ×1.25"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.25)
        decision = {"action": "short", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.BOOST)
        self.assertAlmostEqual(mult, 1.25, places=2)

    def test_inst_sell_long_reduce(self):
        """规则4: 机构卖出 (ratio<0.4) + 做多 → REDUCE ×0.7"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.35)
        decision = {"action": "long", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.REDUCE)
        self.assertAlmostEqual(mult, 0.7, places=2)

    def test_inst_sell_short_boost(self):
        """规则4: 机构卖出 (ratio<0.4) + 做空 → BOOST ×1.15"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.35)
        decision = {"action": "short", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.BOOST)
        self.assertAlmostEqual(mult, 1.15, places=2)

    def test_balanced_allow(self):
        """均衡 → ALLOW"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.50)
        decision = {"action": "long", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.ALLOW)
        self.assertAlmostEqual(mult, 1.0, places=2)

    def test_flat_neutral(self):
        """flat → NEUTRAL"""
        gate = InstitutionRetailGate()
        analysis = make_analysis(0.75)
        decision = {"action": "flat", "quantity": 1.0}
        action, mult, reason = gate._evaluate_rules(analysis, decision)
        self.assertEqual(action, IRAction.NEUTRAL)


class TestApplyToDecision(unittest.TestCase):
    """测试决策门控 (使用 mock)"""

    def test_disabled_gate(self):
        """禁用时直接返回原 decision"""
        gate = InstitutionRetailGate(enabled=False)
        decision = {"action": "long", "quantity": 1.0}
        result = gate.apply_to_decision(decision, symbol="BTC")
        self.assertEqual(result["quantity"], 1.0)
        self.assertNotIn("ir_gate_action", result)
        self.assertEqual(gate._total_calls, 0)

    @patch.object(InstitutionRetailGate, "analyze")
    def test_strong_buy_long_adjustment(self, mock_analyze):
        """强机构买入 + 做多 → BOOST ×1.25"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.75)
        decision = {"action": "long", "quantity": 1.0, "price": 50000}
        result = gate.apply_to_decision(decision, symbol="BTC")
        self.assertEqual(result["ir_gate_action"], "boost")
        self.assertAlmostEqual(result["ir_gate_multiplier"], 1.25, places=2)
        self.assertAlmostEqual(result["quantity"], 1.25, places=2)
        self.assertIn("ir_institution_ratio", result)

    @patch.object(InstitutionRetailGate, "analyze")
    def test_strong_sell_short_adjustment(self, mock_analyze):
        """强机构卖出 + 做空 → BOOST ×1.25"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.25)
        decision = {"action": "short", "quantity": 1.0}
        result = gate.apply_to_decision(decision, symbol="BTC")
        self.assertEqual(result["ir_gate_action"], "boost")
        self.assertAlmostEqual(result["ir_gate_multiplier"], 1.25, places=2)

    @patch.object(InstitutionRetailGate, "analyze")
    def test_balanced_no_adjustment(self, mock_analyze):
        """均衡时不调整"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.50)
        decision = {"action": "long", "quantity": 1.0}
        result = gate.apply_to_decision(decision, symbol="BTC")
        self.assertNotIn("ir_gate_action", result)
        self.assertEqual(result["quantity"], 1.0)

    @patch.object(InstitutionRetailGate, "analyze")
    def test_no_data_no_adjustment(self, mock_analyze):
        """无数据时不调整"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.5, has_data=False)
        decision = {"action": "long", "quantity": 1.0}
        result = gate.apply_to_decision(decision, symbol="BTC")
        self.assertNotIn("ir_gate_action", result)
        self.assertEqual(gate._no_data_count, 1)

    @patch.object(InstitutionRetailGate, "analyze")
    def test_action_variants_long(self, mock_analyze):
        """测试 long/buy 变体"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.75)
        for action in ["long", "buy", "LONG", "BUY"]:
            gate.reset()
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, symbol="BTC")
            self.assertEqual(
                result["ir_gate_action"], "boost",
                f"Failed for action={action}",
            )

    @patch.object(InstitutionRetailGate, "analyze")
    def test_action_variants_short(self, mock_analyze):
        """测试 short/sell 变体"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.75)
        for action in ["short", "sell", "SHORT", "SELL"]:
            gate.reset()
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, symbol="BTC")
            self.assertEqual(
                result["ir_gate_action"], "reduce",
                f"Failed for action={action}",
            )

    @patch.object(InstitutionRetailGate, "analyze")
    def test_zero_quantity(self, mock_analyze):
        """零仓位"""
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.75)
        decision = {"action": "long", "quantity": 0.0}
        result = gate.apply_to_decision(decision, symbol="BTC")
        self.assertEqual(result["quantity"], 0.0)


class TestSyntheticData(unittest.TestCase):
    """测试合成数据生成"""

    def test_generate_synthetic_data(self):
        gate = InstitutionRetailGate()
        data = gate.generate_synthetic_data("BTC")
        self.assertIn("buySellRatio", data)
        self.assertIn("buyVol", data)
        self.assertIn("sellVol", data)
        self.assertIn("timestamp", data)
        buy_vol = float(data["buyVol"])
        sell_vol = float(data["sellVol"])
        self.assertGreater(buy_vol, 0)
        self.assertGreater(sell_vol, 0)
        total = buy_vol + sell_vol
        ratio = buy_vol / total
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_synthetic_data_parseable(self):
        """合成数据可以被 _parse_volume_data 解析"""
        gate = InstitutionRetailGate()
        data = gate.generate_synthetic_data("ETH")
        result = gate._parse_volume_data(data)
        self.assertTrue(result.has_data)
        self.assertGreater(result.total_volume, 0)


class TestStatsAndReset(unittest.TestCase):
    """测试统计和重置"""

    def test_stats_initial(self):
        gate = InstitutionRetailGate()
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["total_adjusted"], 0)
        self.assertEqual(stats["boost_count"], 0)
        self.assertEqual(stats["reduce_count"], 0)
        self.assertEqual(stats["no_data_count"], 0)
        self.assertEqual(stats["adjust_rate"], 0.0)
        self.assertTrue(stats["enabled"])

    @patch.object(InstitutionRetailGate, "analyze")
    def test_stats_after_calls(self, mock_analyze):
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.75)
        decision = {"action": "long", "quantity": 1.0}
        gate.apply_to_decision(decision, symbol="BTC")
        gate.apply_to_decision(decision, symbol="BTC")
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["total_adjusted"], 2)
        self.assertEqual(stats["boost_count"], 2)
        self.assertEqual(stats["reduce_count"], 0)
        self.assertAlmostEqual(stats["adjust_rate"], 1.0, places=2)

    def test_reset(self):
        gate = InstitutionRetailGate()
        gate._total_calls = 10
        gate._total_adjusted = 5
        gate._boost_count = 3
        gate._reduce_count = 2
        gate.reset()
        self.assertEqual(gate._total_calls, 0)
        self.assertEqual(gate._total_adjusted, 0)
        self.assertEqual(gate._boost_count, 0)
        self.assertEqual(gate._reduce_count, 0)


class TestRoundComplete(unittest.TestCase):
    """测试周期完成回调"""

    @patch.object(InstitutionRetailGate, "analyze")
    def test_round_complete(self, mock_analyze):
        gate = InstitutionRetailGate()
        mock_analyze.return_value = make_analysis(0.75)
        decision = {"action": "long", "quantity": 1.0}
        gate.apply_to_decision(decision, symbol="BTC")
        # 不应抛出异常
        gate.round_complete()
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 1)


class TestMultiRoundStability(unittest.TestCase):
    """多轮稳定性"""

    @patch.object(InstitutionRetailGate, "analyze")
    def test_100_rounds_no_crash(self, mock_analyze):
        gate = InstitutionRetailGate()
        ratios = [0.75, 0.25, 0.65, 0.35, 0.50]
        for i in range(100):
            mock_analyze.return_value = make_analysis(ratios[i % len(ratios)])
            action = "long" if i % 2 == 0 else "short"
            decision = {"action": action, "quantity": 1.0}
            result = gate.apply_to_decision(decision, symbol="BTC")
            self.assertIsNotNone(result)
        stats = gate.get_stats()
        self.assertEqual(stats["total_calls"], 100)
        self.assertGreater(stats["total_adjusted"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
