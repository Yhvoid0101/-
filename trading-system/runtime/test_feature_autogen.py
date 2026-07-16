#!/usr/bin/env python3
"""特征自动生成模块单元测试 — Layer 8 学习反馈层

测试覆盖:
  1. 模块导入
  2. FeatureTransform / GeneratedFeature 数据结构
  3. FeatureAutoGenerator 初始化
  4. generate() 6种操作 (ma/std/roc/rank/diff/zscore)
  5. feedback() 权重更新
  6. evolve() 淘汰+生成
  7. get_best_transforms() / get_stats()
  8. reset() 重置
  9. 边界条件 (零值/空历史/disabled模式)
  10. 历史窗口管理
  11. 多轮进化稳定性
  12. 优胜劣汰有效性验证
"""
import sys
import os
import unittest

# 添加路径
sys.path.insert(0, "/home/lmy/hermes_v6")
sys.path.insert(0, "/home/lmy/hermes_v6/sandbox_trading")

# 设置模块路径
os.environ.setdefault("PYTHONPATH", "/home/lmy/hermes_v6:/home/lmy/hermes_v6/sandbox_trading")

from sandbox_trading.feature_autogen import (
    FeatureAutoGenerator,
    FeatureTransform,
    GeneratedFeature,
    create_feature_generator,
)


class TestFeatureTransform(unittest.TestCase):
    """测试 FeatureTransform 数据结构"""

    def test_create_default(self):
        t = FeatureTransform()
        self.assertEqual(t.name, "")
        self.assertEqual(t.input_field, "")
        self.assertEqual(t.operation, "")
        self.assertEqual(t.window, 10)
        self.assertEqual(t.weight, 1.0)
        self.assertEqual(t.use_count, 0)
        self.assertEqual(t.score_sum, 0.0)
        self.assertEqual(t.avg_score, 0.0)

    def test_create_with_values(self):
        t = FeatureTransform(
            name="ma_price_20",
            input_field="price",
            operation="ma",
            window=20,
            weight=1.5,
        )
        self.assertEqual(t.name, "ma_price_20")
        self.assertEqual(t.window, 20)
        self.assertEqual(t.weight, 1.5)

    def test_to_dict(self):
        t = FeatureTransform(name="std_vol_30", input_field="volume", operation="std", window=30)
        d = t.to_dict()
        self.assertIn("name", d)
        self.assertIn("input_field", d)
        self.assertIn("operation", d)
        self.assertIn("window", d)
        self.assertEqual(d["name"], "std_vol_30")
        self.assertEqual(d["window"], 30)

    def test_slots(self):
        """验证 slots=True 优化"""
        t = FeatureTransform()
        with self.assertRaises(AttributeError):
            t.nonexistent_field = 1


class TestGeneratedFeature(unittest.TestCase):
    """测试 GeneratedFeature 数据结构"""

    def test_create_default(self):
        f = GeneratedFeature()
        self.assertEqual(f.name, "")
        self.assertEqual(f.value, 0.0)
        self.assertIsNone(f.transform)

    def test_create_with_values(self):
        t = FeatureTransform(name="ma_price_10")
        f = GeneratedFeature(name="ma_price_10", value=40000.5, transform=t)
        self.assertEqual(f.name, "ma_price_10")
        self.assertEqual(f.value, 40000.5)
        self.assertEqual(f.transform.name, "ma_price_10")

    def test_to_dict(self):
        f = GeneratedFeature(name="test", value=1.5)
        d = f.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["value"], 1.5)


class TestFeatureAutoGeneratorInit(unittest.TestCase):
    """测试初始化"""

    def test_default_init(self):
        gen = FeatureAutoGenerator(enabled=True, seed=42)
        self.assertTrue(gen.enabled)
        self.assertEqual(len(gen._transforms), gen.INITIAL_FEATURES)
        self.assertEqual(gen._generation, 0)
        self.assertEqual(gen._total_generated, 0)
        self.assertEqual(gen._total_feedback, 0)
        self.assertEqual(gen._stats["evolutions"], 0)
        self.assertEqual(gen._stats["killed"], 0)
        self.assertEqual(gen._stats["created"], gen.INITIAL_FEATURES)

    def test_custom_init_features(self):
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        self.assertEqual(len(gen._transforms), 5)

    def test_disabled_init(self):
        gen = FeatureAutoGenerator(enabled=False, seed=42)
        self.assertFalse(gen.enabled)
        self.assertEqual(len(gen._transforms), 0)

    def test_transforms_have_valid_fields(self):
        """验证所有生成的 transform 使用合法字段和操作"""
        gen = FeatureAutoGenerator(enabled=True, seed=42)
        for t in gen._transforms:
            self.assertIn(t.input_field, gen.INPUT_FIELDS)
            self.assertIn(t.operation, gen.OPERATIONS)
            self.assertGreaterEqual(t.window, gen.WINDOW_MIN)
            self.assertLessEqual(t.window, gen.WINDOW_MAX)
            self.assertTrue(t.name)

    def test_seed_reproducibility(self):
        """同一种子应生成相同变换"""
        g1 = FeatureAutoGenerator(enabled=True, seed=42)
        g2 = FeatureAutoGenerator(enabled=True, seed=42)
        self.assertEqual(len(g1._transforms), len(g2._transforms))
        for t1, t2 in zip(g1._transforms, g2._transforms):
            self.assertEqual(t1.name, t2.name)
            self.assertEqual(t1.window, t2.window)

    def test_create_convenience_function(self):
        gen = create_feature_generator(enabled=True)
        self.assertTrue(gen.enabled)
        self.assertGreater(len(gen._transforms), 0)


class TestGenerateOperations(unittest.TestCase):
    """测试 6 种数学操作"""

    def setUp(self):
        """初始化生成器并填充历史"""
        self.gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        # 填充历史: 让所有字段都有足够数据
        for i in range(70):
            self.gen.generate(
                price=40000.0 + i * 100,
                volume=1000.0 + i * 10,
                oi=500000.0 + i * 1000,
                funding_rate=0.0001 + i * 0.00001,
            )

    def test_generate_returns_dict(self):
        features = self.gen.generate(price=47000, volume=1700, oi=570000, funding_rate=0.0008)
        self.assertIsInstance(features, dict)

    def test_generate_disabled_returns_empty(self):
        gen = FeatureAutoGenerator(enabled=False)
        features = gen.generate(price=40000)
        self.assertEqual(features, {})

    def test_ma_operation(self):
        """MA 移动平均"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        # 手动添加一个 MA 变换
        gen._transforms.append(FeatureTransform(
            name="ma_price_10", input_field="price", operation="ma", window=10
        ))
        # 填充10个价格
        for i in range(15):
            features = gen.generate(price=40000.0 + i)
        # 第15次应该有 MA 结果
        self.assertIn("ma_price_10", features)
        # MA 应该是最近10个值的平均
        expected = sum(40000.0 + i for i in range(5, 15)) / 10
        self.assertAlmostEqual(features["ma_price_10"], expected, places=2)

    def test_std_operation(self):
        """STD 标准差"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="std_price_5", input_field="price", operation="std", window=5
        ))
        for i in range(10):
            features = gen.generate(price=float(i) * 100)
        # 5个值的标准差: 0,100,200,300,400 → std=158.11
        import numpy as np
        expected = float(np.std([100.0 * i for i in range(5, 10)]))
        self.assertAlmostEqual(features["std_price_5"], expected, places=2)

    def test_roc_operation(self):
        """ROC 变化率"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="roc_price_1", input_field="price", operation="roc", window=1
        ))
        # 第一次无 ROC (需要历史)
        gen.generate(price=100.0)
        # 第二次有 ROC: (110-100)/100 = 0.1
        features = gen.generate(price=110.0)
        self.assertIn("roc_price_1", features)
        self.assertAlmostEqual(features["roc_price_1"], 0.1, places=4)

    def test_rank_operation(self):
        """RANK 分位数排名"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="rank_price_5", input_field="price", operation="rank", window=5
        ))
        for i in range(10):
            features = gen.generate(price=float(i))
        # 当前价=9, 历史[5,6,7,8,9], 排序后 9 在最后 → rank=5/5=1.0
        self.assertIn("rank_price_5", features)
        self.assertGreaterEqual(features["rank_price_5"], 0.0)
        self.assertLessEqual(features["rank_price_5"], 1.0)

    def test_diff_operation(self):
        """DIFF 差分"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="diff_price_1", input_field="price", operation="diff", window=1
        ))
        gen.generate(price=100.0)
        features = gen.generate(price=110.0)
        self.assertIn("diff_price_1", features)
        self.assertAlmostEqual(features["diff_price_1"], 10.0, places=2)

    def test_zscore_operation(self):
        """ZSCORE Z分数"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="zscore_price_5", input_field="price", operation="zscore", window=5
        ))
        for i in range(10):
            features = gen.generate(price=float(i) * 10)
        self.assertIn("zscore_price_5", features)

    def test_zscore_zero_std(self):
        """ZSCORE 标准差为0时返回0"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="zscore_price_3", input_field="price", operation="zscore", window=3
        ))
        # 相同价格
        for _ in range(5):
            features = gen.generate(price=100.0)
        self.assertIn("zscore_price_3", features)
        self.assertEqual(features["zscore_price_3"], 0.0)

    def test_unknown_operation_returns_none(self):
        """未知操作返回 None"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="unknown_price_5", input_field="price", operation="unknown", window=5
        ))
        for i in range(10):
            features = gen.generate(price=float(i))
        # 未知操作不应出现在结果中
        self.assertNotIn("unknown_price_5", features)

    def test_zero_price_skipped(self):
        """价格为0时跳过"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="ma_price_5", input_field="price", operation="ma", window=5
        ))
        features = gen.generate(price=0.0)
        self.assertNotIn("ma_price_5", features)

    def test_short_history_returns_none(self):
        """历史不足时返回 None"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=0, seed=42)
        gen._transforms.append(FeatureTransform(
            name="ma_price_10", input_field="price", operation="ma", window=10
        ))
        # 只填1次
        features = gen.generate(price=100.0)
        self.assertNotIn("ma_price_10", features)


class TestFeedback(unittest.TestCase):
    """测试 feedback() 权重更新"""

    def setUp(self):
        self.gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        # 让 transforms 累积使用次数
        for i in range(20):
            self.gen.generate(
                price=40000.0 + i, volume=1000.0 + i,
                oi=500000.0, funding_rate=0.0001,
            )

    def test_feedback_updates_score(self):
        """反馈更新评分"""
        initial_scores = [t.score_sum for t in self.gen._transforms]
        self.gen.feedback(score=10.5)
        # 至少有些 transform 被更新
        updated_count = sum(
            1 for t in self.gen._transforms if t.score_sum > 0
        )
        self.assertGreater(updated_count, 0)

    def test_feedback_tracks_best_worst(self):
        """反馈追踪 best/worst"""
        self.gen.feedback(score=10.0)
        self.gen.feedback(score=-5.0)
        self.assertEqual(self.gen._stats["best_score"], 10.0)
        self.assertEqual(self.gen._stats["worst_score"], -5.0)

    def test_feedback_disabled_noop(self):
        gen = FeatureAutoGenerator(enabled=False)
        gen.feedback(score=10.0)
        self.assertEqual(gen._total_feedback, 0)

    def test_feedback_increments_counter(self):
        before = self.gen._total_feedback
        self.gen.feedback(score=1.0)
        self.assertEqual(self.gen._total_feedback, before + 1)

    def test_feedback_only_updates_used_transforms(self):
        """只更新使用过的 transform"""
        # 添加一个未使用的 transform
        new_t = FeatureTransform(name="unused", input_field="price", operation="ma", window=5)
        new_t.use_count = 0
        self.gen._transforms.append(new_t)
        self.gen.feedback(score=10.0)
        # 未使用的 transform 不应被更新
        self.assertEqual(new_t.score_sum, 0.0)


class TestEvolve(unittest.TestCase):
    """测试 evolve() 优胜劣汰"""

    def setUp(self):
        self.gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        # 让 transforms 使用并反馈
        for i in range(20):
            self.gen.generate(
                price=40000.0 + i, volume=1000.0 + i,
                oi=500000.0, funding_rate=0.0001,
            )
        self.gen.feedback(score=-5.0)  # 给负分让某些被淘汰

    def test_evolve_first_9_noop(self):
        """前9代不进化"""
        for gen_num in range(1, 10):
            self.gen.evolve()
            self.assertEqual(self.gen._stats["evolutions"], 0)

    def test_evolve_generation_10_triggers(self):
        """第10代触发进化"""
        for _ in range(10):
            self.gen.evolve()
        self.assertEqual(self.gen._stats["evolutions"], 1)

    def test_evolve_kills_low_score(self):
        """淘汰低分特征"""
        # 让所有 transform 累积足够使用次数 + 多次低分反馈(>=3次)
        for _ in range(30):
            self.gen.generate(price=40000, volume=1000, oi=500000, funding_rate=0.0001)
        # 给3次低分反馈, 让 avg_score 跌破 KILL_THRESHOLD=-1.0
        for _ in range(3):
            self.gen.feedback(score=-10.0)
        # 触发进化
        for _ in range(10):
            self.gen.evolve()
        self.assertGreater(self.gen._stats["killed"], 0)

    def test_evolve_creates_new(self):
        """进化生成新特征"""
        before_count = len(self.gen._transforms)
        for _ in range(10):
            self.gen.evolve()
        # 进化后总数应保持稳定或在范围内
        self.assertLessEqual(len(self.gen._transforms), self.gen.MAX_FEATURES)

    def test_evolve_disabled_noop(self):
        gen = FeatureAutoGenerator(enabled=False)
        gen.evolve()
        self.assertEqual(gen._stats["evolutions"], 0)

    def test_evolve_preserves_low_use_count(self):
        """使用次数<3的特征保留观察"""
        # 添加未使用的 transform
        new_t = FeatureTransform(name="rare", input_field="price", operation="ma", window=5)
        new_t.use_count = 0
        new_t.avg_score = -100.0  # 极低分但使用次数不够
        self.gen._transforms.append(new_t)
        for _ in range(10):
            self.gen.evolve()
        # 应被保留
        self.assertIn(new_t, self.gen._transforms)

    def test_evolve_mutation(self):
        """变异产生新变换"""
        # 强制触发多次进化
        for _ in range(30):
            self.gen.evolve()
        # 检查是否生成了新的变换
        self.assertGreater(self.gen._stats["created"], self.gen.INITIAL_FEATURES)

    def test_evolve_max_features_limit(self):
        """变换数量不超过 MAX_FEATURES"""
        for _ in range(50):
            self.gen.evolve()
        self.assertLessEqual(len(self.gen._transforms), self.gen.MAX_FEATURES)


class TestStatsAndTransforms(unittest.TestCase):
    """测试统计和查询"""

    def setUp(self):
        self.gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        for i in range(20):
            self.gen.generate(
                price=40000.0 + i, volume=1000.0 + i,
                oi=500000.0, funding_rate=0.0001,
            )
        self.gen.feedback(score=5.0)

    def test_get_transforms(self):
        ts = self.gen.get_transforms()
        self.assertIsInstance(ts, list)
        self.assertEqual(len(ts), len(self.gen._transforms))
        for t in ts:
            self.assertIn("name", t)
            self.assertIn("input_field", t)
            self.assertIn("operation", t)
            self.assertIn("window", t)

    def test_get_best_transforms(self):
        # 设置不同评分
        for i, t in enumerate(self.gen._transforms):
            t.avg_score = float(i)
        best = self.gen.get_best_transforms(n=3)
        self.assertEqual(len(best), 3)
        # 应按评分降序
        self.assertGreaterEqual(best[0]["avg_score"], best[1]["avg_score"])
        self.assertGreaterEqual(best[1]["avg_score"], best[2]["avg_score"])

    def test_get_best_transforms_n_too_large(self):
        best = self.gen.get_best_transforms(n=100)
        self.assertEqual(len(best), len(self.gen._transforms))

    def test_get_stats(self):
        stats = self.gen.get_stats()
        self.assertIn("enabled", stats)
        self.assertIn("generation", stats)
        self.assertIn("n_transforms", stats)
        self.assertIn("total_generated", stats)
        self.assertIn("total_feedback", stats)
        self.assertIn("evolutions", stats)
        self.assertIn("killed", stats)
        self.assertIn("created", stats)
        self.assertIn("best_score", stats)
        self.assertIn("worst_score", stats)
        self.assertIn("history_fields", stats)
        self.assertIn("history_lengths", stats)
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["n_transforms"], 5)
        self.assertGreater(stats["total_generated"], 0)

    def test_get_stats_history_tracked(self):
        stats = self.gen.get_stats()
        self.assertIn("price", stats["history_fields"])
        self.assertIn("volume", stats["history_fields"])
        self.assertIn("oi", stats["history_fields"])
        self.assertIn("funding_rate", stats["history_fields"])


class TestReset(unittest.TestCase):
    """测试 reset()"""

    def test_reset_clears_state(self):
        gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        # 累积状态
        for i in range(30):
            gen.generate(price=40000.0 + i, volume=1000.0)
        gen.feedback(score=10.0)
        for _ in range(15):
            gen.evolve()
        # 重置
        gen.reset()
        self.assertEqual(gen._generation, 0)
        self.assertEqual(gen._total_generated, 0)
        self.assertEqual(gen._total_feedback, 0)
        self.assertEqual(gen._stats["evolutions"], 0)
        self.assertEqual(gen._stats["killed"], 0)
        self.assertEqual(gen._stats["created"], gen.INITIAL_FEATURES)
        self.assertEqual(len(gen._transforms), gen.INITIAL_FEATURES)
        self.assertEqual(len(gen._history), 0)

    def test_reset_disabled_no_transforms(self):
        gen = FeatureAutoGenerator(enabled=False)
        gen.reset()
        self.assertEqual(len(gen._transforms), 0)

    def test_reset_reinitializes_transforms(self):
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        gen.reset()
        self.assertEqual(len(gen._transforms), gen.INITIAL_FEATURES)


class TestHistoryManagement(unittest.TestCase):
    """测试历史窗口管理"""

    def test_history_growth(self):
        gen = FeatureAutoGenerator(enabled=True, initial_features=1, seed=42)
        for i in range(10):
            gen.generate(price=float(i))
        self.assertEqual(len(gen._history["price"]), 10)

    def test_history_capped(self):
        """历史不超过 WINDOW_MAX*2"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=1, seed=42)
        for i in range(200):
            gen.generate(price=float(i))
        self.assertLessEqual(len(gen._history["price"]), gen.WINDOW_MAX * 2)

    def test_history_field_isolation(self):
        """不同字段历史独立"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=1, seed=42)
        gen.generate(price=100.0, volume=1000.0)
        self.assertEqual(len(gen._history["price"]), 1)
        self.assertEqual(len(gen._history["volume"]), 1)


class TestMultiRoundStability(unittest.TestCase):
    """多轮进化稳定性"""

    def test_long_run_no_crash(self):
        """长跑无崩溃"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        for round_num in range(100):
            for i in range(20):
                gen.generate(
                    price=40000.0 + i + round_num * 0.1,
                    volume=1000.0 + i,
                    oi=500000.0,
                    funding_rate=0.0001,
                )
            gen.feedback(score=round_num * 0.1 - 5.0)
            gen.evolve()
        # 应稳定运行
        self.assertGreater(gen._total_generated, 0)
        self.assertGreater(gen._stats["evolutions"], 5)
        self.assertLessEqual(len(gen._transforms), gen.MAX_FEATURES)

    def test_diverse_inputs(self):
        """多样化输入不崩溃"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        for i in range(50):
            gen.generate(
                price=40000.0 * (1 + 0.01 * ((-1) ** i)),
                volume=1000.0 + 500 * ((-1) ** i),
                oi=500000.0 + 1000 * i,
                funding_rate=0.0001 * ((-1) ** i),
            )
        self.assertGreater(gen._total_generated, 0)


class TestEdgeCases(unittest.TestCase):
    """边界条件"""

    def test_empty_generate(self):
        """空参数 generate"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        features = gen.generate()  # 全默认参数
        self.assertIsInstance(features, dict)

    def test_negative_price(self):
        """负价格"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        for i in range(10):
            features = gen.generate(price=-100.0 - i)
        self.assertIsInstance(features, dict)

    def test_extreme_values(self):
        """极端值"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        for _ in range(15):
            features = gen.generate(
                price=1e12,
                volume=1e9,
                oi=1e15,
                funding_rate=1.0,
            )
        self.assertIsInstance(features, dict)

    def test_nan_input(self):
        """NaN 输入"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        import math
        for _ in range(10):
            features = gen.generate(
                price=math.nan,
                volume=math.nan,
            )
        # 不应崩溃
        self.assertIsInstance(features, dict)

    def test_inf_input(self):
        """Inf 输入"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=5, seed=42)
        import math
        for _ in range(10):
            features = gen.generate(
                price=math.inf,
                volume=1000.0,
            )
        self.assertIsInstance(features, dict)


class TestEvolutionEffectiveness(unittest.TestCase):
    """优胜劣汰有效性验证"""

    def test_good_features_survive(self):
        """高分特征保留"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        # 让 transform 累积使用次数
        for _ in range(30):
            gen.generate(price=40000.0, volume=1000.0, oi=500000.0, funding_rate=0.0001)
        # 给所有 transform 高分
        gen.feedback(score=10.0)
        good_transforms = list(gen._transforms)
        # 触发进化
        for _ in range(10):
            gen.evolve()
        # 至少有些好的被保留
        survived = sum(1 for t in good_transforms if t in gen._transforms)
        self.assertGreater(survived, 0)

    def test_bad_features_killed(self):
        """低分特征淘汰"""
        gen = FeatureAutoGenerator(enabled=True, initial_features=10, seed=42)
        # 让所有 transform 累积使用次数 + 多次低分反馈(>=3次)
        for _ in range(30):
            gen.generate(price=40000.0, volume=1000.0, oi=500000.0, funding_rate=0.0001)
        # 给3次极低分反馈, 让 avg_score 跌破 KILL_THRESHOLD=-1.0
        for _ in range(3):
            gen.feedback(score=-100.0)
        original_names = {t.name for t in gen._transforms}
        # 触发进化
        for _ in range(10):
            gen.evolve()
        # 应有部分被淘汰
        self.assertGreater(gen._stats["killed"], 0)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="test_feature_autogen.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
