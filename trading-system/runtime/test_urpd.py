import logging
logging.basicConfig(level=logging.DEBUG)
from urpd_realized_price import URPDGate
gate = URPDGate(enabled=True)
# Test with current BTC price
result = gate.analyze(market_price=64142.0)
print(f"market_price={result.market_price}")
print(f"realized_price={result.realized_price}")
print(f"mvrv_ratio={result.mvrv_ratio}")
print(f"level={result.level}")
print(f"has_data={result.has_data}")
print()
# Test apply_to_decision
decision = {"action": "buy", "quantity": 1.0, "price": 64142.0}
new_decision = gate.apply_to_decision(decision, symbol="BTC-USDT", market_price=64142.0)
print(f"action: {new_decision.get('urpd_gate_action')}")
print(f"multiplier: {new_decision.get('urpd_gate_multiplier')}")
print(f"reason: {new_decision.get('urpd_gate_reason')}")
print(f"mvrv: {new_decision.get('urpd_mvrv_ratio')}")
print(f"quantity: {new_decision.get('quantity')}")
print(f"stats: calls={gate._total_calls} adjusted={gate._total_adjusted} boost={gate._boost_count} reduce={gate._reduce_count} no_data={gate._no_data_count}")
