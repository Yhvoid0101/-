import unittest

from sandbox_trading.trade_validation import has_valid_stop_distances


class TradeValidationTest(unittest.TestCase):
    def test_rejects_stop_distances_that_exceed_half_of_entry_price(self):
        self.assertFalse(has_valid_stop_distances(1.455, 61.64, 184.92))

    def test_accepts_atr_scaled_stop_distances(self):
        self.assertTrue(has_valid_stop_distances(1.455, 0.0389, 0.1167))


if __name__ == "__main__":
    unittest.main()