import unittest

from sandbox_trading.data_pipeline import DataPipeline


class IndicatorAlignmentTest(unittest.TestCase):
    def test_indicator_calculation_survives_pipeline_reset_with_aligned_ohlcv(self):
        pipeline = DataPipeline()
        pipeline.initialize(['SOL-USDT'], bars_per_symbol=1000, use_mock=False, use_local_real=True)
        pipeline.reset()
        for _ in range(20):
            tick = pipeline.next_tick('SOL-USDT')
        self.assertGreater(tick['indicators']['atr_14'], 0)
        self.assertGreaterEqual(tick['indicators']['adx_14'], 0)


if __name__ == '__main__':
    unittest.main()