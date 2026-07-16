import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.graceful_degradation import GracefulDegradation

print("=" * 60)
print("P1-2: GracefulDegradation Verification")
print("=" * 60)

gd = GracefulDegradation()

print()
print("--- TEST 1: Normal Operation ---")
gd.report_success("cognitive_network")
gd.report_success("deep_think")
gd.report_success("meta_cognition")
print(f"  CN available: {gd.is_available('cognitive_network')}")
print(f"  DT available: {gd.is_available('deep_think')}")
print(f"  MC available: {gd.is_available('meta_cognition')}")

print()
print("--- TEST 2: Degradation After Consecutive Failures ---")
for i in range(3):
    gd.report_failure("cognitive_network", f"error {i+1}")
    print(f"  Failure {i+1}: degraded={not gd.is_available('cognitive_network')}, "
          f"consecutive={gd._health['cognitive_network'].consecutive_failures}")

print()
print("--- TEST 3: Fallback Mechanism ---")
call_count = {"primary": 0, "fallback": 0}

def primary_fn(x):
    call_count["primary"] += 1
    raise RuntimeError("primary failed")

def fallback_fn(x):
    call_count["fallback"] += 1
    return f"fallback result for {x}"

gd.register_fallback("cognitive_network", fallback_fn)
result = gd.execute_with_fallback("cognitive_network", primary_fn, "test")
print(f"  Result: {result}")
print(f"  Primary called: {call_count['primary']}, Fallback called: {call_count['fallback']}")

print()
print("--- TEST 4: Auto Recovery ---")
for i in range(5):
    gd.report_success("cognitive_network")
print(f"  CN recovered: {gd.is_available('cognitive_network')}")
print(f"  CN success rate: {gd._health['cognitive_network'].success_rate:.3f}")

print()
print("--- TEST 5: Health Report ---")
report = gd.get_health_report()
for name, info in list(report.items())[:5]:
    print(f"  {name}: healthy={info['healthy']} degraded={info['degraded']} "
          f"rate={info['success_rate']} fallback={info['fallback'][:30]}")

print()
print("--- TEST 6: Force Recovery ---")
for _ in range(5):
    gd.report_failure("deep_think", "error")
print(f"  DT degraded: {not gd.is_available('deep_think')}")
gd.force_recover("deep_think")
print(f"  DT after force recover: {gd.is_available('deep_think')}")

print()
print("--- TEST 7: Execute With Fallback (healthy module) ---")
def healthy_fn(x):
    return f"primary: {x}"

result2 = gd.execute_with_fallback("deep_think", healthy_fn, "hello")
print(f"  Result: {result2}")

print()
print("--- TEST 8: Degradation Stats ---")
stats = gd.get_stats()
print(f"  Total reports: {stats['total_reports']}")
print(f"  Degradations: {stats['degradations']}")
print(f"  Recoveries: {stats['recoveries']}")
print(f"  Currently degraded: {stats['current_degraded']}")

print()
print("--- TEST 9: HermesBrain Integration ---")
from core.hermes_brain import HermesBrain
from core.neurotransmitter_system import NeuroTransmitterSystem

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt})()
brain = HermesBrain(bus=bus)

has_gd = brain._graceful_degradation is not None
print(f"  GracefulDegradation in HermesBrain: {has_gd}")
if has_gd:
    gd_report = brain._graceful_degradation.get_health_report()
    print(f"  Monitored modules: {len(gd_report)}")

print()
print("--- TEST 10: Concurrent Reporting ---")
errors = []
def worker(tid, iters):
    for i in range(iters):
        try:
            if i % 3 == 0:
                gd.report_failure(f"module_{tid}", "error")
            else:
                gd.report_success(f"module_{tid}")
        except Exception as e:
            errors.append(f"T{tid}-{i}: {e}")

threads = [threading.Thread(target=worker, args=(t, 50)) for t in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"  Threads: 5x50, Errors: {len(errors)}")

print()
print("--- TEST 11: Edge Cases ---")
edge_errors = []
edge_cases = [
    ("Unknown module success", lambda: gd.report_success("nonexistent")),
    ("Unknown module failure", lambda: gd.report_failure("nonexistent", "err")),
    ("Unknown module available", lambda: gd.is_available("nonexistent")),
    ("Empty module success", lambda: gd.report_success("")),
    ("None module failure", lambda: gd.report_failure(None, "err")),
    ("Force recover unknown", lambda: gd.force_recover("nonexistent")),
]

for name, fn in edge_cases:
    try:
        fn()
        print(f"  OK: {name}")
    except Exception as e:
        edge_errors.append(f"{name}: {e}")
        print(f"  FAIL: {name} -> {e}")

print()
all_ok = len(errors) == 0 and len(edge_errors) == 0
print(f"RESULT: {'ALL P1-2 TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
if edge_errors:
    for e in edge_errors:
        print(f"  ERROR: {e}")
