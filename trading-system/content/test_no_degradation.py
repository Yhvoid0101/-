import pytest
from sandbox_trading.multi_model_ensemble import DynamicWeightManager, LightGBMPredictor, ModelPerformanceMonitor, ProphetTrendAnalyzer, SignalFusionLayer

def test_drifted_model_is_quarantined_and_excluded_from_weights():
    monitor = ModelPerformanceMonitor()
    monitor.detect_performance_drift = lambda name: {"drifted": name == "lightgbm", "early_acc": 0.8, "recent_acc": 0.2}
    manager = DynamicWeightManager(performance_monitor=monitor)
    weights = manager.get_weights(["lightgbm", "prophet"])
    assert "lightgbm" not in weights
    assert "lightgbm" in manager.quarantine
    assert weights["prophet"] == pytest.approx(1.0)

def test_lightgbm_without_model_is_unavailable_without_mock(monkeypatch):
    predictor = LightGBMPredictor()
    monkeypatch.setattr(predictor, "_mock_predict", lambda *_: pytest.fail("mock prediction called"))
    result = predictor.predict("BTCUSDT", {}, [100.0] * 30)
    assert result["model_available"] is False
    assert result["status"] == "unavailable"

def test_lightgbm_inference_error_is_error_without_mock(monkeypatch):
    predictor = LightGBMPredictor()
    predictor._booster = object()
    monkeypatch.setattr(predictor, "_real_predict", lambda *_: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(predictor, "_mock_predict", lambda *_: pytest.fail("mock prediction called"))
    result = predictor.predict("BTCUSDT", {}, [100.0] * 30)
    assert result["model_available"] is False
    assert result["status"] == "error"

def test_prophet_unavailable_is_unavailable_without_mock(monkeypatch):
    analyzer = ProphetTrendAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_prophet_checked", lambda: False)
    monkeypatch.setattr(analyzer, "_mock_prophet", lambda *_: pytest.fail("mock prophet called"))
    result = analyzer.analyze("BTCUSDT", [100.0] * 40)
    assert result["model_available"] is False
    assert result["status"] == "unavailable"

def test_prophet_error_is_unavailable_without_linear_regression_mock(monkeypatch):
    analyzer = ProphetTrendAnalyzer()
    monkeypatch.setattr(analyzer, "_ensure_prophet_checked", lambda: True)
    monkeypatch.setattr(analyzer, "_real_prophet", lambda *_: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(analyzer, "_mock_prophet", lambda *_: pytest.fail("mock prophet called"))
    result = analyzer.analyze("BTCUSDT", [100.0] * 40)
    assert result["model_available"] is False
    assert result["status"] == "error"

def test_fusion_ignores_unavailable_signals_even_if_prediction_fields_exist():
    result = SignalFusionLayer().fuse(
        {"model_available": False, "probability": 0.99, "direction": "long"},
        {"model_available": False, "forecast_price": 101.0, "direction": "long"},
        {"model_available": False, "sentiment_score": 1.0, "direction": "long"},
    )
    assert result["active_models"] == []
    assert result["tradable"] is False
    assert result["status"] == "unavailable"