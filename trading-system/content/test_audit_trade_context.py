from pathlib import Path


def test_open_trade_audit_includes_risk_and_generation_context():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")
    start = source.index('action="open"')
    end = source.index('\n\n                # v2.19 P3', start)
    open_trade_call = source[start:end]

    for field in ('"stop_loss"', '"take_profit"', '"generation"', '"roc_10"', '"primary_signals"'):
        assert field in open_trade_call