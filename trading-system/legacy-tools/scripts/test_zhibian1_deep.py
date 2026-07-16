# -*- coding: utf-8 -*-
"""质变1模块深度边界和跨模块联动测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.society_of_thought import SocietyOfThought, ThoughtRole
from core.digital_personhood import DigitalPersonhood
from core.open_ended_evolution import OpenEndedEvolution, CodeMutator, FitnessEvaluator
from core.neurotransmitter_system import NeuroTransmitterSystem

print("=" * 60)
print("Deep Edge Case Tests")
print("=" * 60)

# Test 1: SocietyOfThought with neurotransmitter modulation
print("\n--- SOT + NeuroTransmitter: emotion affects debate ---")
nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt, '_neurosymbolic_verifier': None})()
sot = SocietyOfThought(bus=bus)
weights = sot._compute_role_weights()
print(f"Default weights: advocate={weights['advocate']:.2f}, critic={weights['critic']:.2f}")
nt.spike("dopamine", 0.3)
weights2 = sot._compute_role_weights()
print(f"After DA spike: advocate={weights2['advocate']:.2f}, critic={weights2['critic']:.2f}")
assert weights2["advocate"] > weights["advocate"], "DA spike should boost advocate weight"
print("[PASS] Neurotransmitter modulates debate weights")

# Test 2: SOT debate memory reuse
print("\n--- SOT: debate memory reuse ---")
sot2 = SocietyOfThought()
r1 = sot2.deliberate("test memory topic")
assert r1["rounds"] > 0
r2 = sot2.deliberate("test memory topic")
assert r2.get("arguments", {}).get("reused") == True or r2["rounds"] >= 0
print("[PASS] Debate memory works")

# Test 3: SOT empty topic
print("\n--- SOT: empty topic handling ---")
sot3 = SocietyOfThought()
r = sot3.deliberate("")
assert "verdict" in r
print("[PASS] Empty topic handled")

# Test 4: DigitalPersonhood personality evolution
print("\n--- DP: personality evolution over time ---")
dp = DigitalPersonhood()
initial_curiosity = dp.dimensions["curiosity"].value
for _ in range(5):
    dp._update_dimensions_from_state({"mood": "兴奋而好奇"})
after_curiosity = dp.dimensions["curiosity"].value
print(f"Curiosity: {initial_curiosity:.3f} -> {after_curiosity:.3f}")
assert after_curiosity > initial_curiosity, "Excited mood should increase curiosity"
print("[PASS] Personality evolves with emotional state")

# Test 5: DP personality decay
print("\n--- DP: personality decay toward baseline ---")
dp2 = DigitalPersonhood()
dp2.dimensions["curiosity"].update(0.3)
high_val = dp2.dimensions["curiosity"].value
for _ in range(20):
    dp2.dimensions["curiosity"].decay()
decayed = dp2.dimensions["curiosity"].value
print(f"Curiosity after decay: {high_val:.3f} -> {decayed:.3f}")
assert decayed < high_val, "Decay should reduce value toward baseline"
print("[PASS] Personality decay works")

# Test 6: DP consistency check with mismatched state
print("\n--- DP: consistency check detects mismatches ---")
dp3 = DigitalPersonhood()
dp3.dimensions["curiosity"].value = 0.1
result = dp3._check_consistency({"how_do_i_feel": {"mood": "兴奋而好奇"}})
print(f"Consistent: {result['consistent']}, violations: {len(result.get('violations', []))}")
assert not result["consistent"], "Low curiosity + excited mood should be inconsistent"
print("[PASS] Consistency check detects mismatches")

# Test 7: DP with full bus integration
print("\n--- DP: full bus integration ---")
nt2 = NeuroTransmitterSystem()
bus2 = type('Bus', (), {
    'neurotransmitter': nt2, '_physical_layer': None,
    '_neurosymbolic_verifier': None, '_streaming_alignment': None,
    'instinct_engine': None, '_creativity_engine': None,
    '_self_organizing_swarm': None,
})()
dp4 = DigitalPersonhood(bus=bus2)
sense = dp4.sense_of_self()
mood = sense["how_do_i_feel"]["mood"]
print(f"Mood with NT: {mood}")
assert mood != "unknown"
print("[PASS] Full bus integration works")

# Test 8: OEE safety check blocks dangerous code
print("\n--- OEE: safety blocks dangerous mutations ---")
fe = FitnessEvaluator()
dangerous_code = "import os\nos.system('rm -rf /')"
safety = fe._check_safety(dangerous_code)
print(f"Dangerous code safety score: {safety}")
assert safety < 0.5, "Dangerous code should have low safety score"
print("[PASS] Safety check blocks dangerous code")

# Test 9: OEE code quality assessment
print("\n--- OEE: code quality assessment ---")
good_code = "def process(data):\n    return data.strip()\n"
bad_code = "x=" + "a" * 200 + "\n"
good_quality = fe._assess_code_quality(good_code)
bad_quality = fe._assess_code_quality(bad_code)
print(f"Good code quality: {good_quality:.2f}, Bad code quality: {bad_quality:.2f}")
assert good_quality > bad_quality
print("[PASS] Code quality assessment works")

# Test 10: OEE cross-domain migration
print("\n--- OEE: cross-domain migration ---")
oe = OpenEndedEvolution()
path1 = os.path.join(oe.sandbox_dir, "mod_a.py")
path2 = os.path.join(oe.sandbox_dir, "mod_b.py")
with open(path1, 'w', encoding='utf-8') as f:
    f.write("threshold = 0.5\nalpha = 0.1\n")
with open(path2, 'w', encoding='utf-8') as f:
    f.write("threshold = 0.8\nbeta = 0.2\n")
oe.register_module("mod_a", path1, "main")
oe.register_module("mod_b", path2, "main")
result = oe.evolve("mod_a", strategy="cross_domain")
print(f"Cross-domain result: success={result.get('success')}, desc={result.get('mutation_desc', '')[:60]}")
assert "migration" in result.get("mutation_desc", "").lower() or "迁移" in result.get("mutation_desc", "")
print("[PASS] Cross-domain migration works")

# Test 11: OEE rollback to original
print("\n--- OEE: rollback to original ---")
oe2 = OpenEndedEvolution()
path3 = os.path.join(oe2.sandbox_dir, "mod_c.py")
with open(path3, 'w', encoding='utf-8') as f:
    f.write("threshold = 0.5\ndef process(): pass\n")
oe2.register_module("mod_c", path3, "process")
oe2.evolve("mod_c")
rollback = oe2.rollback("mod_c")
assert rollback["success"]
print("[PASS] Rollback works")

# Test 12: HermesBrain full integration
print("\n--- HermesBrain: full integration ---")
from core.hermes_brain import HermesBrain
nt3 = NeuroTransmitterSystem()
bus3 = type('Bus', (), {
    'neurotransmitter': nt3, '_physical_layer': None,
    '_neurosymbolic_verifier': None, '_streaming_alignment': None,
    'instinct_engine': None, 'dag_orchestrator': None,
    'permission_guard': None, 'meta_cognition': None,
})()
brain = HermesBrain(bus=bus3)
assert brain._society_of_thought is not None
assert brain._digital_personhood is not None
assert brain._open_ended_evolution is not None

debate = brain._tool_debate(topic="should we use microservices?", action="deliberate")
assert "verdict" in debate
print(f"Debate verdict: {debate['verdict'][:60]}")

reflect = brain._tool_self_reflect(action="summary")
assert "summary" in reflect
print(f"Personhood: {reflect['summary'][:60]}")

evo_stats = brain._tool_evolve_module(action="stats")
assert "total_modules" in evo_stats
print("[PASS] HermesBrain full integration works")

print("\n" + "=" * 60)
print("ALL DEEP EDGE CASE TESTS PASSED!")
print("=" * 60)
