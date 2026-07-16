import unittest

from sandbox_trading.trade_validation import has_auditable_signal_source


class SignalSourceValidationTest(unittest.TestCase):
    def test_rejects_missing_or_none_signal_sources(self):
        self.assertFalse(has_auditable_signal_source(''))
        self.assertFalse(has_auditable_signal_source('none'))
        self.assertFalse(has_auditable_signal_source('final_action=long;none'))

    def test_accepts_named_signal_source(self):
        self.assertTrue(has_auditable_signal_source('macd_cross'))
        self.assertTrue(has_auditable_signal_source('final_action=short;price_momentum_roc'))


if __name__ == '__main__':
    unittest.main()