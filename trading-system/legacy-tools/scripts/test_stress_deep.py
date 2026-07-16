import sys, os, threading, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.deep_think import DeepThink
from core.meta_cognition import MetaCognition
from core.neurotransmitter_system import NeuroTransmitterSystem
from core.predictive_engine import PredictiveEngine
from core.cognitive_network import CognitiveNetwork
from core.attention_memory import AttentionMemory
from core.temporal_integration import TemporalIntegration
from core.cognitive_topology import CognitiveTopology
from core.memory_crystallizer import MemoryCrystallizer
from core.instinct_engine import InstinctEngine
from core.deep_causal_reasoner import DeepCausalReasoner

errors = []

def setup_bus():
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, 'modules': {},
    })()
    dt = DeepThink(bus=bus)
    mc = MetaCognition(bus=bus)
    pe = PredictiveEngine(bus=bus)
    cn = CognitiveNetwork(bus=bus)
    am = AttentionMemory(bus=bus)
    ti = TemporalIntegration(bus=bus)
    ct = CognitiveTopology(bus=bus)
    ie = InstinctEngine(bus=bus)
    dcr = DeepCausalReasoner(bus=bus)
    mc2 = MemoryCrystallizer(bus=bus)
    bus.deep_think = dt
    bus.meta_cognition = mc
    bus.predictive_engine = pe
    bus.cognitive_network = cn
    bus.attention_memory = am
    bus.temporal_integration = ti
    bus.cognitive_topology = ct
    bus.instinct_engine = ie
    bus.deep_causal_reasoner = dcr
    bus.memory_crystallizer = mc2
    return bus, dt, mc, pe, cn, am, ti, ct, ie, dcr, mc2

print("=" * 60)
print("STRESS TEST 1: 50 Rapid Feedback Cycles")
print("=" * 60)

bus, dt, mc, pe, cn, am, ti, ct, ie, dcr, mc2 = setup_bus()

for i in range(50):
    try:
        if i % 2 == 0:
            bus.neurotransmitter.record_result(True, 0.9, 'positive')
        else:
            bus.neurotransmitter.record_result(False, 0.2, 'negative')

        result = {
            'task': f'cycle_{i}',
            'plan': {'steps': [
                {'action': f'step_a_{i}', 'success': i % 3 != 0},
                {'action': f'step_b_{i}', 'success': True, 'output': f'result_{i}'},
            ]},
            'result': {'success': i % 5 != 0},
            'evaluation': {'total_score': 0.5 + (i % 5) * 0.1}
        }
        reflection = mc.reflect(result)

        dt.receive_bias_corrections(reflection.get('improvement_suggestions', []))

        pred = pe.predict("fix" if i % 2 == 0 else "research")
        pe.resolve_prediction(pred, i % 3 == 0, "high" if i % 4 == 0 else "low")

        am.allocate_attention(task=f"task_{i}", task_type="fix" if i % 2 == 0 else "research")

        ct.record_tool_usage("analyze_code" if i % 3 == 0 else "deep_research")

        dcr.trace_causal_chain(f"chain_{i}", [{'detail': f'event_{j}'} for j in range(3)])

    except Exception as e:
        errors.append(f"Cycle {i}: {e}")
        traceback.print_exc()

print(f"  Cycles: 50, Errors: {len(errors)}")
print(f"  NT levels: DA={bus.neurotransmitter.dopamine:.2f} NE={bus.neurotransmitter.norepinephrine:.2f}")
print(f"  DeepThink corrections: {len(dt._recent_bias_corrections)}")
print(f"  TI attention weights: {len(ti._attention_weights)}")
print(f"  DCR chains: {len(dcr.causal_chains)}")
print(f"  CT activations: {sum(ct._tool_activation.values())}")
if errors:
    for e in errors[:5]:
        print(f"  ERROR: {e}")

print()
print("=" * 60)
print("STRESS TEST 2: Concurrent Multi-Loop Firing")
print("=" * 60)

errors2 = []

def worker_loop(thread_id, iterations):
    for i in range(iterations):
        try:
            pred = pe.predict("fix")
            pe.resolve_prediction(pred, i % 2 == 0, "medium")
            am.allocate_attention(task=f"t{thread_id}_{i}", task_type="fix")
            ct.record_tool_usage("analyze_code")
        except Exception as e:
            errors2.append(f"T{thread_id}-{i}: {e}")

threads = [threading.Thread(target=worker_loop, args=(t, 20)) for t in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"  Threads: 5, Iterations each: 20, Errors: {len(errors2)}")
if errors2:
    for e in errors2[:5]:
        print(f"  ERROR: {e}")

print()
print("=" * 60)
print("STRESS TEST 3: Extreme Neurotransmitter Values")
print("=" * 60)

errors3 = []

for label, setup_fn in [
    ("All success x20", lambda: [bus.neurotransmitter.record_result(True, 1.0, 'positive') for _ in range(20)]),
    ("All failure x20", lambda: [bus.neurotransmitter.record_result(False, 0.0, 'negative') for _ in range(20)]),
]:
    nt2 = NeuroTransmitterSystem()
    bus2 = type('Bus', (), {'neurotransmitter': nt2, 'modules': {}})()
    dt2 = DeepThink(bus=bus2)
    bus2.deep_think = dt2
    setup_fn()
    try:
        levels = nt2.get_levels()
        thought = dt2.think(f"test under {label}")
        prompt = dt2._build_enhanced_prompt("test", thought)
        print(f"  {label}: DA={levels['dopamine']:.2f} NE={levels['norepinephrine']:.2f} -> think OK, solutions={len(thought.proposed_solutions)}")
    except Exception as e:
        errors3.append(f"{label}: {e}")
        print(f"  {label}: FAILED - {e}")

print(f"  Extreme value errors: {len(errors3)}")

print()
print("=" * 60)
print("STRESS TEST 4: Empty/None/Invalid Inputs")
print("=" * 60)

errors4 = []
bus3, dt3, mc3, pe3, cn3, am3, ti3, ct3, ie3, dcr3, mc3_2 = setup_bus()

edge_cases = [
    ("Empty think", lambda: dt3.think("")),
    ("Whitespace think", lambda: dt3.think("   ")),
    ("Very long input", lambda: dt3.think("x" * 5000)),
    ("Unicode input", lambda: dt3.think("修复🔥代码🚀问题")),
    ("Empty reflect", lambda: mc3.reflect({'task': '', 'plan': {'steps': []}, 'result': {}, 'evaluation': {}})),
    ("None reflect fields", lambda: mc3.reflect({'task': None, 'plan': None, 'result': None, 'evaluation': None})),
    ("Empty predict", lambda: pe3.predict("")),
    ("Empty causal chain", lambda: dcr3.trace_causal_chain("", [])),
    ("None causal chain", lambda: dcr3.trace_causal_chain(None, None)),
    ("Empty attention", lambda: am3.allocate_attention(task="", task_type="")),
    ("Empty temporal event", lambda: ti3.add_raw_event("", "")),
    ("Negative importance", lambda: ti3.add_raw_event("test", "detail", importance=-1.0)),
    ("Importance > 1", lambda: ti3.add_raw_event("test", "detail", importance=5.0)),
    ("Empty topology query", lambda: ct3.find_nearest_tool("", top_n=0)),
    ("Receive None corrections", lambda: dt3.receive_bias_corrections(None)),
    ("Receive empty corrections", lambda: dt3.receive_bias_corrections([])),
    ("Modulate cluster empty", lambda: ie3.modulate_cluster("", 1.0)),
    ("Modulate cluster extreme", lambda: ie3.modulate_cluster("explore", 100.0)),
    ("Instinct influence empty", lambda: am3.apply_instinct_influence({})),
    ("Instinct influence None values", lambda: am3.apply_instinct_influence({"risk": None})),
]

for name, fn in edge_cases:
    try:
        result = fn()
        print(f"  OK: {name} -> {type(result).__name__}")
    except Exception as e:
        errors4.append(f"{name}: {e}")
        print(f"  FAIL: {name} -> {e}")

print(f"  Edge case errors: {len(errors4)}")

print()
print("=" * 60)
print("STRESS TEST 5: State Consistency After 100 Cycles")
print("=" * 60)

bus4, dt4, mc4, pe4, cn4, am4, ti4, ct4, ie4, dcr4, mc4_2 = setup_bus()

for i in range(100):
    bus4.neurotransmitter.record_result(i % 3 != 0, 0.5 + (i % 5) * 0.1,
                                         'positive' if i % 3 != 0 else 'negative')
    result = {
        'task': f'task_{i}',
        'plan': {'steps': [{'action': f's{i}', 'success': i % 2 == 0}]},
        'result': {'success': i % 3 != 0},
        'evaluation': {'total_score': 0.5 + (i % 5) * 0.1}
    }
    mc4.reflect(result)
    pe4.predict("fix")
    am4.allocate_attention(task=f"t{i}", task_type="fix")
    ct4.record_tool_usage("analyze_code")

nt_levels = bus4.neurotransmitter.get_levels()
da = nt_levels['dopamine']
ne = nt_levels['norepinephrine']
da_ok = 0.0 <= da <= 1.0
ne_ok = 0.0 <= ne <= 1.0

topo_dist = ct4.concept_distance("修复", "analyze_code")
topo_ok = 0.0 <= topo_dist <= 1.0

cn_health = cn4.get_health_report()
cn_ok = cn_health['health'] in ('good', 'degraded')

dcr_stats = dcr4.get_stats()
dcr_ok = dcr_stats['total_chains'] >= 0

print(f"  DA={da:.3f} (valid={da_ok}), NE={ne:.3f} (valid={ne_ok})")
print(f"  Topo distance={topo_dist:.3f} (valid={topo_ok})")
print(f"  CN health={cn_health['health']} (ok={cn_ok})")
print(f"  DCR chains={dcr_stats['total_chains']} (ok={dcr_ok})")

all_consistent = da_ok and ne_ok and topo_ok and cn_ok and dcr_ok
print(f"  State consistency: {'PASS' if all_consistent else 'FAIL'}")

print()
print("=" * 60)
print("STRESS TEST 6: Feedback Loop Signal Propagation")
print("=" * 60)

bus5, dt5, mc5, pe5, cn5, am5, ti5, ct5, ie5, dcr5, mc5_2 = setup_bus()

initial_risk_weight = am5._attention_slots.get('risk')
initial_risk_val = initial_risk_weight.weight if initial_risk_weight else 0

for _ in range(5):
    bus5.neurotransmitter.record_result(False, 0.1, 'negative')

result_fail = {
    'task': 'dangerous task',
    'plan': {'steps': [
        {'action': 'risky_op', 'success': False, 'output': 'failed error'},
        {'action': 'another_risky', 'success': False, 'output': 'also failed error'},
    ]},
    'result': {'success': False},
    'evaluation': {'total_score': 0.95}
}
reflection = mc5.reflect(result_fail)

corrections_in_dt = len(dt5._recent_bias_corrections)
prompt = dt5._build_enhanced_prompt("dangerous task", dt5.think("dangerous task"))
has_correction = "偏差修正提醒" in prompt

pred5 = pe5.predict("fix")
pe5.resolve_prediction(pred5, False, "critical")

weights5 = am5.allocate_attention(task="dangerous task", task_type="security")
ti_received = len(ti5._attention_weights) > 0

print(f"  Corrections received by DT: {corrections_in_dt}")
print(f"  Correction in prompt: {has_correction}")
print(f"  TI received attention: {ti_received}")
print(f"  Signal propagation: {'PASS' if corrections_in_dt > 0 and has_correction and ti_received else 'FAIL'}")

print()
print("=" * 60)
total_errors = len(errors) + len(errors2) + len(errors3) + len(errors4)
print(f"TOTAL ERRORS ACROSS ALL STRESS TESTS: {total_errors}")
if total_errors > 0:
    print("ERRORS FOUND - NEED FIXING:")
    all_errs = errors + errors2 + errors3 + errors4
    for e in all_errs[:20]:
        print(f"  {e}")
else:
    print("ALL STRESS TESTS PASSED - ZERO ERRORS")
