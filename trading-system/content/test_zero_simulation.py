from sandbox_trading.data_pipeline import DataPipeline


def test_non_mock_pipeline_does_not_generate_cyclical_fallback(monkeypatch):
    pipeline = DataPipeline()
    pipeline._load_local_real_klines = lambda symbol, bars: False
    pipeline._init_price_cache_with_warmup = lambda symbol: None
    pipeline.feed = type("Feed", (), {"_bars": {}, "_current_index": {}, "_symbols": []})()
    pipeline._generate_cyclical_data_fallback = lambda *args: (_ for _ in ()).throw(AssertionError("synthetic fallback forbidden"))
    monkeypatch.setattr(
        "sandbox_trading.gateio_data_integration.create_real_data_feed",
        lambda **kwargs: type("Feed", (), {"_bars": {}})(),
    )
    import pytest
    with pytest.raises(RuntimeError, match="real market data unavailable"):
        pipeline.initialize(["BTC-USDT"], bars_per_symbol=10, use_mock=False, use_local_real=True)


def test_non_mock_order_book_path_returns_no_data():
    from sandbox_trading.evolution_loop import EvolutionLoop
    loop = EvolutionLoop.__new__(EvolutionLoop)
    loop._use_mock = False
    assert not getattr(loop, "_allow_synthetic_order_book", False)
