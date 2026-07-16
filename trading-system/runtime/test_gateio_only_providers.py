from sandbox_trading.multi_exchange_provider import MultiExchangeAggregator
from sandbox_trading.real_exchange_feed import RealExchangeFeed


def test_aggregator_default_has_no_binance_provider():
    provider = MultiExchangeAggregator(enable_mexc=False, enable_bitget=False, enable_bybit=False)
    assert [item.exchange_name for item in provider._providers] == ["Gate.io"]


def test_real_feed_default_uses_gate_only():
    feed = RealExchangeFeed()
    assert feed.vision is None
    assert feed.futures is None
    assert feed.gate is not None


def test_real_feed_does_not_call_binance(monkeypatch):
    feed = RealExchangeFeed()
    feed.gate.fetch_ticker = lambda symbol: None
    feed.cache.load_ticker = lambda symbol: None
    assert feed.fetch_ticker("BTCUSDT") is None
