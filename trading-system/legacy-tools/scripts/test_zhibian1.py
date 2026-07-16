# -*- coding: utf-8 -*-
"""质变1模块综合终端测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback

PASS = 0
FAIL = 0

def test(name, func):
    global PASS, FAIL
    try:
        func()
        PASS += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name}: {str(e)[:80]}")
        traceback.print_exc()

print("=" * 60)
print("Phase 1: SocietyOfThought Tests")
print("=" * 60)

from core.society_of_thought import SocietyOfThought, ThoughtRole, DebateMemory

def test_sot_init():
    sot = SocietyOfThought()
    assert len(sot.roles) == 5
    stats = sot.get_stats()
    assert "total_debates" in stats

def test_sot_local_deliberate():
    sot = SocietyOfThought()
    result = sot.deliberate("修复长函数时应该优先使用AST拆分还是手动重构？")
    assert "verdict" in result
    assert result["confidence"] > 0
    assert len(result["arguments"]["supporting"]) > 0
    assert len(result["arguments"]["opposing"]) > 0

def test_sot_role_subset():
    sot = SocietyOfThought()
    result = sot.deliberate("验证数据是否正确", role_subset=["advocate", "critic", "arbitrator"])
    assert "verdict" in result
    assert "explorer" not in result.get("active_roles", [])

def test_sot_convergence():
    sot = SocietyOfThought(rounds=3)
    result = sot.deliberate("简单的是否问题")
    assert "converged" in result

def test_sot_memory():
    dm = DebateMemory()
    dm.store("test topic", {"verdict": "test", "confidence": 0.9, "converged": True})
    similar = dm.find_similar("test topic")
    assert len(similar) >= 1

def test_sot_local_reasoning():
    role = ThoughtRole("advocate")
    result = role.deliberate("修复bug")
    assert result["role"] == "advocate"
    assert result["source"] == "local"
    assert len(result["content"]) > 10

def test_sot_critic_local():
    role = ThoughtRole("critic")
    result = role.deliberate("删除所有文件")
    assert "风险" in result["content"] or "不可逆" in result["content"]

for t in [test_sot_init, test_sot_local_deliberate, test_sot_role_subset,
          test_sot_convergence, test_sot_memory, test_sot_local_reasoning,
          test_sot_critic_local]:
    test(t.__name__, t)

print(f"\nSocietyOfThought: {PASS} passed, {FAIL} failed")
p1, f1 = PASS, FAIL

print("\n" + "=" * 60)
print("Phase 2: DigitalPersonhood Tests")
print("=" * 60)

from core.digital_personhood import DigitalPersonhood, PersonalityDimension, GrowthTracker

def test_dp_init():
    dp = DigitalPersonhood()
    assert len(dp.dimensions) == 6
    stats = dp.get_stats()
    assert "personhood_summary" in stats

def test_dp_sense_of_self():
    dp = DigitalPersonhood()
    sense = dp.sense_of_self()
    assert "who_am_i" in sense
    assert "how_do_i_feel" in sense
    assert "what_do_i_want" in sense
    assert "personality" in sense

def test_dp_personality_dimension():
    dim = PersonalityDimension("test_dim", 0.5, 0.5, 0.01)
    dim.update(0.1)
    assert dim.value > 0.5
    dim.decay()
    assert dim.value < 0.6

def test_dp_personality_trend():
    dim = PersonalityDimension("test_dim", 0.5)
    for i in range(10):
        dim.update(0.02)
    trend = dim.get_trend()
    assert trend == "rising"

def test_dp_reflect():
    dp = DigitalPersonhood()
    dp._last_reflection_time = 0
    result = dp.reflect()
    assert "sense" in result
    assert "consistency" in result

def test_dp_summary():
    dp = DigitalPersonhood()
    summary = dp.get_personhood_summary()
    assert len(summary) > 20
    assert "Hermes" in summary

def test_dp_growth_tracker():
    gt = GrowthTracker()
    gt.record("tasks_completed", 10)
    gt.record("tasks_completed", 50)
    summary = gt.get_growth_summary()
    assert "metrics" in summary

def test_dp_consistency_check():
    dp = DigitalPersonhood()
    sense = dp.sense_of_self()
    result = dp._check_consistency(sense)
    assert "consistent" in result

def test_dp_with_neurotransmitter():
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {'neurotransmitter': nt})()
    dp = DigitalPersonhood(bus=bus)
    sense = dp.sense_of_self()
    mood = sense.get("how_do_i_feel", {}).get("mood", "")
    assert mood != "unknown"

for t in [test_dp_init, test_dp_sense_of_self, test_dp_personality_dimension,
          test_dp_personality_trend, test_dp_reflect, test_dp_summary,
          test_dp_growth_tracker, test_dp_consistency_check,
          test_dp_with_neurotransmitter]:
    test(t.__name__, t)

print(f"\nDigitalPersonhood: {PASS - p1} passed, {FAIL - f1} failed")
p2, f2 = PASS, FAIL

print("\n" + "=" * 60)
print("Phase 3: OpenEndedEvolution Tests")
print("=" * 60)

from core.open_ended_evolution import OpenEndedEvolution, CodeMutator, FitnessEvaluator

def test_oee_init():
    oe = OpenEndedEvolution()
    stats = oe.get_stats()
    assert "total_modules" in stats

def test_oee_register():
    oe = OpenEndedEvolution()
    result = oe.register_module("test_mod", "nonexistent.py", "test_func")
    assert result == True

def test_oee_parameter_mutation():
    cm = CodeMutator()
    code = "threshold = 0.5\nalpha = 0.1\nbeta = 0.3"
    new_code, desc = cm.mutate_parameters(code, intensity=0.2)
    assert "threshold" in new_code

def test_oee_fitness_syntax():
    fe = FitnessEvaluator()
    score = fe._check_syntax("x = 1 + 2")
    assert score == 1.0
    score = fe._check_syntax("def (")
    assert score == 0.0

def test_oee_fitness_safety():
    fe = FitnessEvaluator()
    safe = fe._check_safety("x = 1 + 2")
    assert safe > 0.5
    dangerous = fe._check_safety("os.system('rm -rf /')")
    assert dangerous < 0.5

def test_oee_evolve():
    oe = OpenEndedEvolution()
    dummy_path = os.path.join(oe.sandbox_dir, "dummy_evo.py")
    with open(dummy_path, 'w', encoding='utf-8') as f:
        f.write("threshold = 0.5\nalpha = 0.1\ndef process(): pass\n")
    oe.register_module("dummy_evo", dummy_path, "process")
    result = oe.evolve("dummy_evo", strategy="parameter_mutation")
    assert "success" in result
    assert "fitness" in result

def test_oee_rollback():
    oe = OpenEndedEvolution()
    dummy_path = os.path.join(oe.sandbox_dir, "dummy_rb.py")
    with open(dummy_path, 'w', encoding='utf-8') as f:
        f.write("threshold = 0.5\ndef process(): pass\n")
    oe.register_module("dummy_rb", dummy_path, "process")
    result = oe.rollback("dummy_rb")
    assert result["success"] == True

def test_oee_strategy_selection():
    oe = OpenEndedEvolution()
    from core.open_ended_evolution import EvolvableModule
    module = EvolvableModule("test", "test.py", "func", "x=1")
    strategy = oe._select_strategy(module)
    assert strategy in ["parameter_mutation", "logic_mutation",
                        "structural_mutation", "cross_domain", "hybrid"]

for t in [test_oee_init, test_oee_register, test_oee_parameter_mutation,
          test_oee_fitness_syntax, test_oee_fitness_safety, test_oee_evolve,
          test_oee_rollback, test_oee_strategy_selection]:
    test(t.__name__, t)

print(f"\nOpenEndedEvolution: {PASS - p2} passed, {FAIL - f2} failed")
p3, f3 = PASS, FAIL

print("\n" + "=" * 60)
print("Phase 4: HermesBrain Integration Tests")
print("=" * 60)

def test_brain_integration():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    assert brain._society_of_thought is not None
    assert brain._digital_personhood is not None
    assert brain._open_ended_evolution is not None

def test_tool_debate():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    r = brain._tool_debate(topic="test debate topic", action="deliberate")
    assert "verdict" in r

def test_tool_self_reflect():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    r = brain._tool_self_reflect(action="summary")
    assert "summary" in r

def test_tool_evolve():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    r = brain._tool_evolve_module(action="stats")
    assert "total_modules" in r

for t in [test_brain_integration, test_tool_debate,
          test_tool_self_reflect, test_tool_evolve]:
    test(t.__name__, t)

print(f"\nIntegration: {PASS - p3} passed, {FAIL - f3} failed")

print("\n" + "=" * 60)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL PASSED!")
else:
    print("ISSUES FOUND")
