import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.decision_arbiter import DecisionArbiter
arb = DecisionArbiter()
try:
    r = arb.arbitrate({'explore': 0.5}, {'biases_detected': None})
    print(f"OK: {r}")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
