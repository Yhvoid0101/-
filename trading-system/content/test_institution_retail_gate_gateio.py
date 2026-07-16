from sandbox_trading.institution_retail_gate import InstitutionRetailGate


def test_gateio_failure_returns_no_data_without_binance(monkeypatch):
    gate = InstitutionRetailGate(enabled=True)
    gate.fetch_from_gateio = lambda symbol: None
    gate.load_from_cache = lambda symbol: None
    gate.fetch_from_api = lambda symbol: (_ for _ in ()).throw(AssertionError("Binance path must not exist"))
    assert gate.get_volume_data("BTC-USDT") is None


def test_gateio_data_is_used_without_binance(monkeypatch):
    gate = InstitutionRetailGate(enabled=True)
    payload = {"buyVol": "60", "sellVol": "40", "timestamp": "1700000000000"}
    gate.fetch_from_gateio = lambda symbol: payload
    gate.fetch_from_api = lambda symbol: (_ for _ in ()).throw(AssertionError("Binance path must not exist"))
    assert gate.get_volume_data("BTC-USDT") == payload


def test_prefetched_gateio_data_has_priority(monkeypatch):
    gate = InstitutionRetailGate(enabled=True)
    payload = {"buyVol": "60", "sellVol": "40", "timestamp": "1700000000000"}
    monkeypatch.setattr(
        "sandbox_trading.institution_retail_gate.PrefetchedDataReader",
        type("Reader", (), {"read_taker_ratio": staticmethod(lambda symbol: payload)}),
    )
    gate.fetch_from_gateio = lambda symbol: (_ for _ in ()).throw(AssertionError("network should not be called"))
    assert gate.get_volume_data("BTC-USDT") == payload
