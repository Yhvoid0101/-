import unittest

from sandbox_trading.sandbox_engine import MarketData, OrderType, SandboxMatchingEngine


class SandboxMatchingEngineTest(unittest.TestCase):
    def test_zero_cost_round_trip_returns_reserved_margin(self):
        engine = SandboxMatchingEngine(
            maker_fee=0,
            taker_fee=0,
            min_slippage=0,
            max_slippage=0,
            latency_ms=0,
            mev_probability=0,
        )
        market = MarketData(
            symbol="BTC-USDT",
            timestamp=1,
            open=100,
            high=100,
            low=100,
            close=100,
            volume=1_000_000,
            bid=100,
            ask=100,
            bid_depth=1_000_000,
            ask_depth=1_000_000,
        )
        engine.update_market(market)
        engine.submit_long("agent", "BTC-USDT", 10, OrderType.MARKET)
        engine.close_position("agent", "BTC-USDT", OrderType.MARKET)

        self.assertEqual(engine.get_balance("agent"), 10_000)

    def test_forced_take_profit_fill_cannot_cross_below_long_entry(self):
        engine = SandboxMatchingEngine(
            initial_capital=10_000,
            maker_fee=0,
            taker_fee=0,
            min_slippage=0,
            max_slippage=0,
        )
        engine.update_market(MarketData(
            symbol="BTC-USDT", timestamp=1, open=100, high=100, low=100,
            close=100, volume=1_000, bid=100, ask=100,
            bid_depth=1_000, ask_depth=1_000,
        ))
        engine.submit_long("agent", "BTC-USDT", 1, OrderType.MARKET)
        trade = engine.get_positions("agent")["BTC-USDT"]
        trade.forced_exit_price = 110
        engine.update_market(MarketData(
            symbol="BTC-USDT", timestamp=2, open=20, high=111, low=19,
            close=100, volume=1_000, bid=100, ask=100,
            bid_depth=1_000, ask_depth=1_000,
        ))

        order = engine.close_position("agent", "BTC-USDT", OrderType.MARKET)

        self.assertIsNotNone(order)
        closed = engine.get_trades("agent")[-1]
        self.assertGreaterEqual(closed.exit_price, closed.entry_price)


if __name__ == "__main__":
    unittest.main()