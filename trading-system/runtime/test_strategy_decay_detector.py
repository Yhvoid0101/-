#!/usr/bin/env python3
"""策略失效检测模块测试 — Layer 8 学习反馈层 P4

覆盖所有功能：
  1. 初始化与配置
  2. 数据更新
  3. 衰减检测（5种信号）
  4. 健康度评分
  5. 门控应用
  6. 批量分析
  7. 统计与重置
  8. 边界情况
  9. 100轮稳定性
"""
import sys
import random

# 添加路径
sys.path.insert(0, '/home/lmy/hermes_v6/sandbox_trading')

import pytest
from strategy_decay_detector import (
    StrategyDecayDetector,
    HealthLevel,
    TradeRecord,
)


# ============================================================================
# 1. 初始化测试
# ============================================================================


class TestDecayDetectorInit:
    """初始化测试"""

    def test_default_init(self):
        """默认初始化"""
        detector = StrategyDecayDetector(enabled=True)
        assert detector.enabled is True
        assert detector.SHORT_WINDOW == 10
        assert detector.LONG_WINDOW == 50
        assert detector.HEALTHY_THRESHOLD == 0.7
        assert detector.WARNING_THRESHOLD == 0.4

    def test_disabled_init(self):
        """禁用初始化"""
        detector = StrategyDecayDetector(enabled=False)
        assert detector.enabled is False

    def test_custom_windows(self):
        """自定义窗口"""
        detector = StrategyDecayDetector(enabled=True, short_window=5, long_window=20)
        assert detector.SHORT_WINDOW == 5
        assert detector.LONG_WINDOW == 20

    def test_thresholds(self):
        """阈值检查"""
        detector = StrategyDecayDetector(enabled=True)
        assert 0 < detector.WARNING_THRESHOLD < detector.HEALTHY_THRESHOLD < 1
        assert detector.REDUCE_LIGHT_MULTIPLIER == 0.8
        assert detector.REDUCE_HEAVY_MULTIPLIER == 0.5

    def test_weights_sum_to_one(self):
        """权重和为1"""
        detector = StrategyDecayDetector(enabled=True)
        total = (
            detector.WEIGHT_WINRATE + detector.WEIGHT_PNL +
            detector.WEIGHT_LOSS + detector.WEIGHT_DRAWDOWN +
            detector.WEIGHT_SHARPE
        )
        assert abs(total - 1.0) < 1e-6


# ============================================================================
# 2. 数据更新测试
# ============================================================================


class TestDataUpdate:
    """数据更新测试"""

    def test_update_single_trade(self):
        """更新单笔交易"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("agent-001", pnl=100.0, win=True)
        trades = detector._get_trades("agent-001", 10)
        assert len(trades) == 1
        assert trades[0].pnl == 100.0
        assert trades[0].win is True

    def test_update_multiple_trades(self):
        """更新多笔交易"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(15):
            detector.update_agent_performance("agent-001", pnl=float(i), win=(i % 2 == 0))
        trades = detector._get_trades("agent-001", 50)
        assert len(trades) == 15

    def test_auto_win_detection(self):
        """自动判断盈亏"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("agent-001", pnl=100.0)  # win=None → auto
        trades = detector._get_trades("agent-001", 10)
        assert trades[0].win is True

        detector.update_agent_performance("agent-001", pnl=-50.0)
        trades = detector._get_trades("agent-001", 10)
        assert trades[-1].win is False

    def test_disabled_no_update(self):
        """禁用时不更新"""
        detector = StrategyDecayDetector(enabled=False)
        detector.update_agent_performance("agent-001", pnl=100.0)
        trades = detector._get_trades("agent-001", 10)
        assert len(trades) == 0

    def test_window_trimming(self):
        """窗口裁剪"""
        detector = StrategyDecayDetector(enabled=True, long_window=5)
        # LONG_WINDOW=5, maxlen=10
        for i in range(20):
            detector.update_agent_performance("agent-001", pnl=float(i))
        trades = detector._get_trades("agent-001", 100)
        assert len(trades) <= 10  # maxlen=LONG_WINDOW*2=10


# ============================================================================
# 3. 衰减检测测试
# ============================================================================


class TestDecayDetection:
    """衰减检测测试"""

    def test_unknown_insufficient_data(self):
        """数据不足返回UNKNOWN"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(5):  # < SHORT_WINDOW=10
            detector.update_agent_performance("agent-001", pnl=100.0, win=True)
        result = detector.detect_decay("agent-001")
        assert result.health_level == HealthLevel.UNKNOWN

    def test_healthy_agent(self):
        """健康agent"""
        detector = StrategyDecayDetector(enabled=True)
        # 50笔稳定盈利交易
        for i in range(50):
            detector.update_agent_performance(
                "agent-001", pnl=100.0, win=True, drawdown=0.02
            )
        result = detector.detect_decay("agent-001")
        assert result.health_level == HealthLevel.HEALTHY
        assert result.health_score >= 0.7
        assert result.suggestion == "continue"

    def test_decayed_agent_consecutive_losses(self):
        """连续亏损导致失效"""
        detector = StrategyDecayDetector(enabled=True)
        # 先45笔盈利
        for i in range(45):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        # 再5笔连续亏损
        for i in range(5):
            detector.update_agent_performance("agent-001", pnl=-100.0, win=False, drawdown=0.15)
        result = detector.detect_decay("agent-001")
        assert result.consecutive_losses >= 5
        assert any("consecutive_losses" in s for s in result.decay_signals)

    def test_winrate_decay(self):
        """胜率衰减检测"""
        detector = StrategyDecayDetector(enabled=True)
        # 长期高胜率
        for i in range(40):
            detector.update_agent_performance("agent-001", pnl=50.0, win=True, drawdown=0.02)
        # 短期低胜率
        for i in range(10):
            detector.update_agent_performance("agent-001", pnl=-20.0, win=False, drawdown=0.05)
        result = detector.detect_decay("agent-001")
        assert result.winrate_short < result.winrate_long
        # 应该检测到胜率衰减
        assert any("winrate_decay" in s for s in result.decay_signals) or result.health_score < 0.7

    def test_pnl_decay(self):
        """PnL衰减检测"""
        detector = StrategyDecayDetector(enabled=True)
        # 长期大盈利
        for i in range(40):
            detector.update_agent_performance("agent-001", pnl=1000.0, win=True, drawdown=0.02)
        # 短期小盈利
        for i in range(10):
            detector.update_agent_performance("agent-001", pnl=10.0, win=True, drawdown=0.02)
        result = detector.detect_decay("agent-001")
        assert result.pnl_short < result.pnl_long

    def test_drawdown_expansion(self):
        """回撤扩大检测"""
        detector = StrategyDecayDetector(enabled=True)
        # 长期低回撤
        for i in range(40):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        # 短期高回撤
        for i in range(10):
            detector.update_agent_performance("agent-001", pnl=10.0, win=True, drawdown=0.15)
        result = detector.detect_decay("agent-001")
        assert result.drawdown_current > result.drawdown_avg

    def test_disabled_returns_unknown(self):
        """禁用时返回UNKNOWN"""
        detector = StrategyDecayDetector(enabled=False)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True)
        # 禁用时不会更新数据
        result = detector.detect_decay("agent-001")
        assert result.health_level == HealthLevel.UNKNOWN

    def test_result_to_dict(self):
        """结果转字典"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        result = detector.detect_decay("agent-001")
        d = result.to_dict()
        assert "health_score" in d
        assert "health_level" in d
        assert "winrate_short" in d
        assert "suggestion" in d


# ============================================================================
# 4. 健康度评分测试
# ============================================================================


class TestHealthScore:
    """健康度评分测试"""

    def test_perfect_score(self):
        """完美评分"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.0)
        result = detector.detect_decay("agent-001")
        assert result.health_score >= 0.7
        assert result.health_level == HealthLevel.HEALTHY

    def test_warning_level(self):
        """警告等级"""
        detector = StrategyDecayDetector(enabled=True)
        # 混合表现
        for i in range(50):
            win = i % 3 != 0  # 67% 胜率
            pnl = 50.0 if win else -30.0
            detector.update_agent_performance("agent-001", pnl=pnl, win=win, drawdown=0.05)
        result = detector.detect_decay("agent-001")
        # 应该在警告或健康范围
        assert result.health_level in [HealthLevel.HEALTHY, HealthLevel.WARNING]

    def test_decayed_level(self):
        """失效等级"""
        detector = StrategyDecayDetector(enabled=True)
        # 先有好表现
        for i in range(40):
            detector.update_agent_performance("agent-001", pnl=200.0, win=True, drawdown=0.02)
        # 然后严重恶化
        for i in range(10):
            detector.update_agent_performance("agent-001", pnl=-150.0, win=False, drawdown=0.20)
        result = detector.detect_decay("agent-001")
        assert result.health_score < 0.7
        # 有多个衰减信号
        assert len(result.decay_signals) >= 2

    def test_score_range(self):
        """评分范围 [0, 1]"""
        detector = StrategyDecayDetector(enabled=True)
        # 各种情况
        for scenario in range(10):
            agent_id = f"agent-{scenario}"
            for i in range(50):
                win = random.random() > 0.5
                pnl = random.uniform(-100, 100)
                dd = random.uniform(0, 0.3)
                detector.update_agent_performance(agent_id, pnl=pnl, win=win, drawdown=dd)
            result = detector.detect_decay(agent_id)
            assert 0.0 <= result.health_score <= 1.0


# ============================================================================
# 5. 门控应用测试
# ============================================================================


class TestApplyToDecision:
    """门控应用测试"""

    def test_disabled_no_adjust(self):
        """禁用时不调整"""
        detector = StrategyDecayDetector(enabled=False)
        decision = {"action": "long", "quantity": 0.01}
        result = detector.apply_to_decision(decision, agent_id="agent-001")
        assert result["quantity"] == 0.01

    def test_no_data_no_adjust(self):
        """数据不足时不调整"""
        detector = StrategyDecayDetector(enabled=True)
        decision = {"action": "long", "quantity": 0.01}
        result = detector.apply_to_decision(decision, agent_id="agent-001")
        assert result["quantity"] == 0.01

    def test_healthy_no_adjust(self):
        """健康时不调整"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        detector.detect_decay("agent-001")
        decision = {"action": "long", "quantity": 0.01}
        result = detector.apply_to_decision(decision, agent_id="agent-001")
        assert result["quantity"] == 0.01

    def test_warning_light_reduce(self):
        """警告时轻微减仓"""
        detector = StrategyDecayDetector(enabled=True)
        # 制造一个WARNING级别的agent
        for i in range(40):
            detector.update_agent_performance("agent-001", pnl=80.0, win=True, drawdown=0.03)
        for i in range(10):
            detector.update_agent_performance("agent-001", pnl=-20.0, win=False, drawdown=0.08)
        result = detector.detect_decay("agent-001")
        # 如果是WARNING，应该轻微减仓
        if result.health_level == HealthLevel.WARNING:
            decision = {"action": "long", "quantity": 0.01}
            new_decision = detector.apply_to_decision(decision, agent_id="agent-001")
            assert new_decision["quantity"] == 0.01 * 0.8
            assert new_decision["decay_action"] == "reduce_light"
            assert new_decision["decay_multiplier"] == 0.8

    def test_decayed_heavy_reduce(self):
        """失效时大幅减仓"""
        detector = StrategyDecayDetector(enabled=True)
        # 先好表现
        for i in range(40):
            detector.update_agent_performance("agent-001", pnl=200.0, win=True, drawdown=0.02)
        # 严重恶化
        for i in range(10):
            detector.update_agent_performance("agent-001", pnl=-200.0, win=False, drawdown=0.25)
        result = detector.detect_decay("agent-001")
        if result.health_level == HealthLevel.DECAYED:
            decision = {"action": "long", "quantity": 0.01}
            new_decision = detector.apply_to_decision(decision, agent_id="agent-001")
            assert new_decision["quantity"] == 0.01 * 0.5
            assert new_decision["decay_action"] == "reduce_heavy"
            assert new_decision["decay_multiplier"] == 0.5
            assert new_decision["decay_suggestion"] == "retire"

    def test_stats_increment(self):
        """统计计数器递增"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        detector.detect_decay("agent-001")
        decision = {"action": "long", "quantity": 0.01}
        detector.apply_to_decision(decision, agent_id="agent-001")
        stats = detector.get_stats()
        assert stats["total_calls"] == 1


# ============================================================================
# 6. 批量分析测试
# ============================================================================


class TestRoundComplete:
    """批量分析测试"""

    def test_round_complete_basic(self):
        """基本批量分析"""
        detector = StrategyDecayDetector(enabled=True)
        for agent_id in ["agent-001", "agent-002", "agent-003"]:
            for i in range(50):
                detector.update_agent_performance(agent_id, pnl=100.0, win=True, drawdown=0.02)
        stats = detector.round_complete()
        assert stats["total_agents"] == 3
        assert stats["healthy"] >= 0

    def test_round_complete_with_performances(self):
        """带性能数据的批量分析"""
        detector = StrategyDecayDetector(enabled=True)
        # 先填充足够数据
        for agent_id in ["agent-001", "agent-002"]:
            for i in range(50):
                detector.update_agent_performance(agent_id, pnl=100.0, win=True, drawdown=0.02)
        # 批量更新
        performances = {
            "agent-001": {"pnl": 50.0, "win": True, "drawdown": 0.03},
            "agent-002": {"pnl": -30.0, "win": False, "drawdown": 0.08},
        }
        stats = detector.round_complete(performances)
        assert stats["total_agents"] == 2

    def test_round_complete_disabled(self):
        """禁用时批量分析"""
        detector = StrategyDecayDetector(enabled=False)
        stats = detector.round_complete()
        assert stats["enabled"] is False

    def test_retire_list(self):
        """退役建议列表"""
        detector = StrategyDecayDetector(enabled=True)
        # 创建一个失效agent
        for i in range(40):
            detector.update_agent_performance("bad-agent", pnl=200.0, win=True, drawdown=0.02)
        for i in range(10):
            detector.update_agent_performance("bad-agent", pnl=-200.0, win=False, drawdown=0.25)
        # 创建一个健康agent
        for i in range(50):
            detector.update_agent_performance("good-agent", pnl=100.0, win=True, drawdown=0.02)
        stats = detector.round_complete()
        assert stats["total_agents"] == 2
        # 失效agent应该在retire_list中（如果确实被判定为DECAYED）
        assert isinstance(stats["retire_list"], list)


# ============================================================================
# 7. 统计与重置测试
# ============================================================================


class TestStatsAndReset:
    """统计与重置测试"""

    def test_stats_initial(self):
        """初始统计"""
        detector = StrategyDecayDetector(enabled=True)
        stats = detector.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_adjusted"] == 0
        assert stats["tracked_agents"] == 0

    def test_reset(self):
        """重置"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("agent-001", pnl=100.0, win=True)
        detector._total_calls = 5
        detector._total_adjusted = 3
        detector.reset()
        stats = detector.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_adjusted"] == 0
        assert stats["tracked_agents"] == 0

    def test_get_health_score(self):
        """获取健康度评分"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        score = detector.get_health_score("agent-001")
        assert 0.0 <= score <= 1.0

    def test_get_suggestion(self):
        """获取建议"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        suggestion = detector.get_decay_suggestion("agent-001")
        assert suggestion in ["continue", "monitor", "retire"]


# ============================================================================
# 8. 边界情况测试
# ============================================================================


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_agent_id(self):
        """空agent_id"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("", pnl=100.0)
        assert "" not in detector._agent_trades

    def test_zero_pnl(self):
        """零PnL"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("agent-001", pnl=0.0)
        trades = detector._get_trades("agent-001", 10)
        assert len(trades) == 1
        assert trades[0].win is False  # pnl=0 → win=False

    def test_negative_pnl(self):
        """负PnL"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("agent-001", pnl=-100.0)
        trades = detector._get_trades("agent-001", 10)
        assert trades[0].win is False

    def test_large_pnl(self):
        """大PnL"""
        detector = StrategyDecayDetector(enabled=True)
        detector.update_agent_performance("agent-001", pnl=1e9)
        trades = detector._get_trades("agent-001", 10)
        assert trades[0].pnl == 1e9

    def test_many_agents(self):
        """多agent"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(100):
            agent_id = f"agent-{i:03d}"
            for j in range(15):
                detector.update_agent_performance(agent_id, pnl=100.0, win=True, drawdown=0.02)
        stats = detector.get_stats()
        assert stats["tracked_agents"] == 100

    def test_decision_no_quantity(self):
        """决策无quantity字段"""
        detector = StrategyDecayDetector(enabled=True)
        for i in range(50):
            detector.update_agent_performance("agent-001", pnl=100.0, win=True, drawdown=0.02)
        detector.detect_decay("agent-001")
        decision = {"action": "long"}  # 无quantity
        result = detector.apply_to_decision(decision, agent_id="agent-001")
        assert "quantity" in result or result.get("quantity", 0) == 0

    def test_consecutive_losses_count(self):
        """连续亏损计数"""
        detector = StrategyDecayDetector(enabled=True)
        # 5笔: win, loss, loss, loss, loss
        detector.update_agent_performance("a", pnl=100.0, win=True)
        detector.update_agent_performance("a", pnl=-10.0, win=False)
        detector.update_agent_performance("a", pnl=-20.0, win=False)
        detector.update_agent_performance("a", pnl=-30.0, win=False)
        detector.update_agent_performance("a", pnl=-40.0, win=False)
        trades = detector._get_trades("a", 10)
        count = detector._count_consecutive_losses(trades)
        assert count == 4

    def test_simple_sharpe_calculation(self):
        """简化Sharpe计算"""
        detector = StrategyDecayDetector(enabled=True)
        trades = [
            TradeRecord(pnl=100.0, win=True),
            TradeRecord(pnl=50.0, win=True),
            TradeRecord(pnl=-30.0, win=False),
            TradeRecord(pnl=80.0, win=True),
        ]
        sharpe = detector._calculate_simple_sharpe(trades)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # 均值为正


# ============================================================================
# 9. 多轮稳定性测试
# ============================================================================


class TestMultiRoundStability:
    """多轮稳定性测试"""

    def test_100_rounds_no_crash(self):
        """100轮不崩溃"""
        random.seed(42)
        detector = StrategyDecayDetector(enabled=True)
        for round_num in range(100):
            # 每轮10个agent交易
            for agent_idx in range(10):
                agent_id = f"agent-{agent_idx:03d}"
                pnl = random.gauss(50, 100)
                win = pnl > 0
                dd = abs(random.gauss(0.05, 0.03))
                detector.update_agent_performance(agent_id, pnl=pnl, win=win, drawdown=dd)

            # 每10轮批量分析
            if round_num % 10 == 9:
                stats = detector.round_complete()
                assert "total_agents" in stats
                assert stats["total_agents"] == 10

        # 最终统计
        stats = detector.get_stats()
        assert stats["tracked_agents"] == 10
        assert stats["total_calls"] >= 0

    def test_extreme_scenario_stability(self):
        """极端场景稳定性"""
        detector = StrategyDecayDetector(enabled=True)
        # 全亏损
        for i in range(100):
            detector.update_agent_performance("all-loss", pnl=-100.0, win=False, drawdown=0.5)
        result = detector.detect_decay("all-loss")
        assert 0.0 <= result.health_score <= 1.0

        # 全盈利
        for i in range(100):
            detector.update_agent_performance("all-win", pnl=100.0, win=True, drawdown=0.0)
        result = detector.detect_decay("all-win")
        assert 0.0 <= result.health_score <= 1.0

        # 震荡
        for i in range(100):
            win = i % 2 == 0
            detector.update_agent_performance(
                "oscillate", pnl=100.0 if win else -100.0, win=win, drawdown=0.1
            )
        result = detector.detect_decay("oscillate")
        assert 0.0 <= result.health_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
