from pathlib import Path


def test_large_order_slice_failure_does_not_fallback_to_full_market_order():
    source = Path(__file__).with_name("evolution_loop.py").read_text(encoding="utf-8")
    sliced_execution = source[source.index("if _is_large_order_p3:"):source.index("else:\n                            # 小单", source.index("if _is_large_order_p3:"))]

    assert "切片执行失败, 回退到市价单" not in sliced_execution
    assert "回退市价单" not in sliced_execution
    assert "self.engine.submit_long" not in sliced_execution
    assert "self.engine.submit_short" not in sliced_execution