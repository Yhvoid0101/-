from sandbox_trading.population_fitness import FitnessScore


def test_risk_rejections_reduce_an_otherwise_identical_gt_score():
    baseline = FitnessScore(
        sharpe_ratio=2.0,
        win_rate=0.65,
        profit_factor=2.0,
        max_drawdown=0.05,
        total_trades=30,
        total_pnl=100.0,
        calmar_ratio=3.0,
        stability=0.8,
        sortino_ratio=2.5,
        ev_per_trade=0.5,
    )
    rejected = FitnessScore(
        sharpe_ratio=2.0,
        win_rate=0.65,
        profit_factor=2.0,
        max_drawdown=0.05,
        total_trades=30,
        total_pnl=100.0,
        calmar_ratio=3.0,
        stability=0.8,
        sortino_ratio=2.5,
        ev_per_trade=0.5,
        risk_rejection_count=2,
    )

    assert rejected.compute_gt_score() < baseline.compute_gt_score()


def test_risk_rejections_never_improve_a_negative_gt_score():
    baseline = FitnessScore(total_trades=0)
    rejected = FitnessScore(total_trades=0, risk_rejection_count=1)

    assert rejected.compute_gt_score() < baseline.compute_gt_score()