from pathlib import Path


def test_large_order_partial_fill_is_explicitly_audited():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")

    assert "large_order_partial_fill_accepted" in source
    assert '"requested_quantity": decision["quantity"]' in source
    assert '"filled_quantity": _large_exec_result_p3.filled_quantity' in source
    assert '"remaining_quantity": (' in source