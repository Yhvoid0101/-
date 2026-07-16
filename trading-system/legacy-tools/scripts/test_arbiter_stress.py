import sys, os, threading, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.decision_arbiter import DecisionArbiter
from core.neurotransmitter_system import NeuroTransmitterSystem

errors = []

print("=" * 60)
print("P1-1 STRESS TEST 1: 100 Rapid Arbitrations")
print("=" * 60)

arbiter = DecisionArbiter()
for i in range(100):
    try:
        weights = {'explore': 0.3 + i * 0.007, 'avoid': 0.9 - i * 0.007,
                    'alert': 0.5, 'create': 0.4, 'conserve': 0.3}
        biases = [['恐慌偏误'], ['过度自信偏误'], ['确认偏误'], ['惯性偏误'],
                  ['忽视风险'], ['锚定偏误']][i % 6]
        meta = {'biases_detected': biases, 'confidence': 0.5 + (i % 5) * 0.1}
        arbiter.arbitrate(weights, meta)
    except Exception as e:
        errors.append(f"Cycle {i}: {e}")

print(f"  Cycles: 100, Errors: {len(errors)}")
stats = arbiter.get_stats()
print(f"  Total: {stats['total_arbitrations']}, Corrections: {stats['bias_corrections']}")

print()
print("=" * 60)
print("P1-1 STRESS TEST 2: Concurrent Arbitration")
print("=" * 60)

arbiter2 = DecisionArbiter()
errors2 = []

def worker(tid, iters):
    for i in range(iters):
        try:
            w = {'explore': 0.5, 'avoid': 0.5, 'alert': 0.5}
            m = {'biases_detected': ['恐慌偏误'], 'confidence': 0.7}
            arbiter2.arbitrate(w, m)
        except Exception as e:
            errors2.append(f"T{tid}-{i}: {e}")

threads = [threading.Thread(target=worker, args=(t, 30)) for t in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"  Threads: 5x30, Errors: {len(errors2)}")

print()
print("=" * 60)
print("P1-1 STRESS TEST 3: Edge Cases")
print("=" * 60)

edge_cases = [
    ("Empty weights", {}, {'biases_detected': ['恐慌偏误']}),
    ("Empty biases", {'explore': 0.5}, {'biases_detected': []}),
    ("None biases", {'explore': 0.5}, {'biases_detected': None}),
    ("None confidence", {'explore': 0.5}, {'biases_detected': ['恐慌偏误'], 'confidence': None}),
    ("Unknown bias", {'explore': 0.5}, {'biases_detected': ['未知偏误类型']}),
    ("Empty meta", {'explore': 0.5}, {}),
    ("None meta", {'explore': 0.5}, None),
    ("All zero weights", {'explore': 0.0, 'avoid': 0.0}, {'biases_detected': ['恐慌偏误']}),
    ("Very high weights", {'explore': 10.0, 'avoid': 10.0}, {'biases_detected': ['过度自信']}),
    ("Negative weights", {'explore': -0.5, 'avoid': -0.3}, {'biases_detected': ['恐慌偏误']}),
    ("Veto with None action", {'explore': 0.5}, {'biases_detected': []}, None),
    ("Veto safe action", {'explore': 0.5}, {'biases_detected': []}, "正常修复"),
    ("Veto dangerous action", {'explore': 0.5}, {'biases_detected': []}, "删除生产数据"),
    ("Multiple biases", {'explore': 0.8, 'avoid': 0.8, 'alert': 0.5, 'create': 0.7, 'conserve': 0.6},
     {'biases_detected': ['恐慌偏误', '过度自信偏误', '确认偏误'], 'confidence': 0.9}),
]

for case in edge_cases:
    name = case[0]
    weights = case[1]
    meta = case[2] if len(case) > 2 else {}
    action = case[3] if len(case) > 3 else None
    try:
        result = arbiter.arbitrate(weights, meta or {}, proposed_action=action)
        print(f"  OK: {name}")
    except Exception as e:
        errors.append(f"{name}: {e}")
        print(f"  FAIL: {name} -> {e}")

print()
print("=" * 60)
print("P1-1 STRESS TEST 4: Neurotransmitter-Aware")
print("=" * 60)

nt_stress = NeuroTransmitterSystem()
for _ in range(10):
    nt_stress.record_result(False, 0.1, 'negative')
bus_stress = type('Bus', (), {'neurotransmitter': nt_stress})()
arb_s = DecisionArbiter(bus=bus_stress)

nt_calm = NeuroTransmitterSystem()
for _ in range(10):
    nt_calm.record_result(True, 0.9, 'positive')
bus_calm = type('Bus', (), {'neurotransmitter': nt_calm})()
arb_c = DecisionArbiter(bus=bus_calm)

weights_test = {'explore': 0.5, 'avoid': 0.5, 'alert': 0.5}
meta_test = {'biases_detected': ['忽视风险'], 'confidence': 0.8}

result_stress = arb_s.arbitrate(dict(weights_test), meta_test)
result_calm = arb_c.arbitrate(dict(weights_test), meta_test)

print(f"  Stress NE high: explore {weights_test['explore']:.2f} -> {result_stress.get('explore', 0):.3f}")
print(f"  Calm DA high:   explore {weights_test['explore']:.2f} -> {result_calm.get('explore', 0):.3f}")
print(f"  Stress alert:   {weights_test['alert']:.2f} -> {result_stress.get('alert', 0):.3f}")
print(f"  Calm alert:     {weights_test['alert']:.2f} -> {result_calm.get('alert', 0):.3f}")

stress_lower = result_stress.get('explore', 1.0) < result_calm.get('explore', 0.0)
alert_higher = result_stress.get('alert', 0.0) > result_calm.get('alert', 0.0)
print(f"  Stress suppresses explore more: {stress_lower}")
print(f"  Stress boosts alert more: {alert_higher}")

print()
print("=" * 60)
print("P1-1 STRESS TEST 5: HermesBrain Integration")
print("=" * 60)

from core.hermes_brain import HermesBrain

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt})()
brain = HermesBrain(bus=bus)

has_arbiter = brain._decision_arbiter is not None
print(f"  DecisionArbiter in HermesBrain: {has_arbiter}")

if has_arbiter:
    arb_stats = brain._decision_arbiter.get_stats()
    print(f"  Arbiter stats: {arb_stats['total_arbitrations']} arbitrations")

print()
total_errors = len(errors) + len(errors2)
print(f"TOTAL ERRORS: {total_errors}")
if total_errors > 0:
    for e in errors[:10]:
        print(f"  {e}")
else:
    print("ALL P1-1 STRESS TESTS PASSED - ZERO ERRORS")
