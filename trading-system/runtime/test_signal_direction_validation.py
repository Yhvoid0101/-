import unittest

from sandbox_trading.trade_validation import has_directionally_consistent_signal_source


class SignalDirectionValidationTest(unittest.TestCase):
    def test_rejects_action_that_conflicts_with_signal_directions(self):
        self.assertFalse(
            has_directionally_consistent_signal_source(
                'short',
                'price_momentum_roc:long;engulfing_bull:long',
            )
        )
        self.assertFalse(
            has_directionally_consistent_signal_source(
                'long',
                'timesfm_prediction:short;willr_signal:short',
            )
        )

    def test_accepts_action_when_all_signal_directions_agree(self):
        self.assertTrue(
            has_directionally_consistent_signal_source(
                'long',
                'macd_cross:long;trend_pullback:long',
            )
        )
        self.assertTrue(
            has_directionally_consistent_signal_source(
                'short',
                'willr_signal:short;price_momentum_roc:short',
            )
        )


if __name__ == '__main__':
    unittest.main()