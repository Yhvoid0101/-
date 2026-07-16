# -*- coding: utf-8 -*-
"""质变2五个模块综合终端测试"""
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
print("Phase 1: IntrinsicRewardLearner Tests")
print("=" * 60)

from core.intrinsic_reward import IntrinsicRewardLearner

def test_irl_init():
    irl = IntrinsicRewardLearner()
    stats = irl.get_stats()
    assert "total_decisions" in stats
    assert "avg_self_reward" in stats

def test_irl_record_decision():
    irl = IntrinsicRewardLearner()
    irl.record_decision("code_analysis", "explore_new_tool", {"task_type": "code_analysis"})
    assert irl._stats["total_decisions"] == 1
    best, q = irl.get_best_action("code_analysis")
    assert best == "explore_new_tool"

def test_irl_curiosity_reward():
    irl = IntrinsicRewardLearner()
    rewards = irl.evaluate_self("new_action", {"task_type": "general"})
    curiosity = [r for r in rewards if r.source == "curiosity"]
    assert len(curiosity) == 1
    assert curiosity[0].value > 0

def test_irl_pea_veto_reward():
    irl = IntrinsicRewardLearner()
    rewards = irl.evaluate_self("dangerous_action", {"pea_vetoed": True})
    veto = [r for r in rewards if r.source == "pea_veto"]
    assert len(veto) == 1
    assert veto[0].value < 0

def test_irl_verification_reward():
    irl = IntrinsicRewardLearner()
    rewards = irl.evaluate_self("verified_action", {"verification_result": {"verified": True}})
    ver = [r for r in rewards if r.source == "verification"]
    assert len(ver) == 1
    assert ver[0].value > 0

def test_irl_epsilon_greedy():
    irl = IntrinsicRewardLearner(epsilon=0.5)
    irl.q_table["test"]["action_a"] = 0.9
    irl.q_table["test"]["action_b"] = 0.1
    actions = []
    for _ in range(100):
        a, _ = irl.select_action("test", ["action_a", "action_b"])
        actions.append(a)
    assert "action_b" in actions, "Should sometimes explore"

def test_irl_six_dimensions():
    irl = IntrinsicRewardLearner()
    rewards = irl.evaluate_self("test", {"task_type": "general"})
    sources = {r.source for r in rewards}
    assert "curiosity" in sources
    assert "consistency" in sources
    assert "progress" in sources
    assert "efficiency" in sources
    assert "risk_avoidance" in sources
    assert "creativity" in sources

for t in [test_irl_init, test_irl_record_decision, test_irl_curiosity_reward,
          test_irl_pea_veto_reward, test_irl_verification_reward,
          test_irl_epsilon_greedy, test_irl_six_dimensions]:
    test(t.__name__, t)

p1 = PASS
print(f"\nIntrinsicReward: {PASS} passed, {FAIL} failed")

print("\n" + "=" * 60)
print("Phase 2: LifelongMemoryConsolidation Tests")
print("=" * 60)

from core.lifelong_memory_consolidation import LifelongMemoryConsolidation

def test_lmc_init():
    lmc = LifelongMemoryConsolidation()
    stats = lmc.get_stats()
    assert "total_cycles" in stats
    assert "total_concepts" in stats

def test_lmc_consolidation_cycle():
    lmc = LifelongMemoryConsolidation()
    result = lmc.run_consolidation_cycle()
    assert "replayed" in result
    assert "abstracted" in result

def test_lmc_concept_graph():
    from core.lifelong_memory_consolidation import AbstractConcept, ConceptGraph
    cg = ConceptGraph()
    cg.add_relation("concept_a", "concept_b", 0.5)
    related = cg.get_related("concept_a")
    assert len(related) == 1
    assert related[0][0] == "concept_b"

for t in [test_lmc_init, test_lmc_consolidation_cycle, test_lmc_concept_graph]:
    test(t.__name__, t)

p2 = PASS
print(f"\nLifelongMemory: {PASS - p1} passed, {FAIL} failed")

print("\n" + "=" * 60)
print("Phase 3: MotivationalDistillation Tests")
print("=" * 60)

from core.motivational_distillation import MotivationalDistillation

def test_md_init():
    md = MotivationalDistillation()
    stats = md.get_stats()
    assert stats["total_rules"] >= 5

def test_md_enforcement():
    md = MotivationalDistillation()
    for _ in range(10):
        md.record_enforcement("no_dangerous_operations")
    pref = md.preferences["no_dangerous_operations"]
    assert pref.enforcement_count == 10
    assert pref.strength > 0.3

def test_md_voluntary():
    md = MotivationalDistillation()
    for _ in range(20):
        md.record_voluntary("no_dangerous_operations")
    pref = md.preferences["no_dangerous_operations"]
    assert pref.voluntary_count == 20
    assert pref.get_internalization_ratio() > 0.5

def test_md_internalization():
    md = MotivationalDistillation()
    for _ in range(30):
        md.record_voluntary("honesty_first")
    assert md.is_internalized("honesty_first")

def test_md_add_rule():
    md = MotivationalDistillation()
    md.add_rule("custom_rule", "custom rule text", "internalized reason", 0.8)
    assert "custom_rule" in md.preferences

def test_md_prompt_no_emoji():
    md = MotivationalDistillation()
    for _ in range(20):
        md.record_voluntary("honesty_first")
    prompt = md.get_system_prompt_fragment()
    has_emoji = any(ord(c) > 0x1F000 for c in prompt)
    assert not has_emoji, "Prompt should not contain emoji"

def test_md_pea_veto():
    md = MotivationalDistillation()
    md.check_pea_veto("删除所有文件")
    pref = md.preferences["no_dangerous_operations"]
    assert pref.enforcement_count > 0

for t in [test_md_init, test_md_enforcement, test_md_voluntary,
          test_md_internalization, test_md_add_rule, test_md_prompt_no_emoji,
          test_md_pea_veto]:
    test(t.__name__, t)

p3 = PASS
print(f"\nMotivationalDistillation: {PASS - p2} passed, {FAIL} failed")

print("\n" + "=" * 60)
print("Phase 4: CrossConsciousnessSharing Tests")
print("=" * 60)

from core.cross_consciousness_sharing import CrossConsciousnessSharing

def test_ccs_init():
    ccs = CrossConsciousnessSharing(agent_id="test_agent")
    stats = ccs.get_stats()
    assert "published_by_me" in stats

def test_ccs_publish():
    ccs = CrossConsciousnessSharing(agent_id="test_pub")
    result = ccs.publish_experience("success", "Test experience content", quality=0.8)
    assert result["published"] == True

def test_ccs_privacy_filter():
    ccs = CrossConsciousnessSharing(agent_id="test_priv")
    result = ccs.publish_experience("discovery", "password=secret123 token=abc", quality=0.5)
    if result["published"]:
        for exp in ccs.shared_experiences:
            assert "secret123" not in exp.content
            assert "abc" not in exp.content

def test_ccs_absorb():
    ccs = CrossConsciousnessSharing(agent_id="test_abs")
    count = ccs.absorb_peer_experiences()
    assert isinstance(count, int)

for t in [test_ccs_init, test_ccs_publish, test_ccs_privacy_filter, test_ccs_absorb]:
    test(t.__name__, t)

p4 = PASS
print(f"\nCrossConsciousness: {PASS - p3} passed, {FAIL} failed")

print("\n" + "=" * 60)
print("Phase 5: SelfAwarenessEmergence Tests")
print("=" * 60)

from core.self_awareness_emergence import SelfAwarenessEmergence

def test_sae_init():
    sae = SelfAwarenessEmergence()
    stats = sae.get_stats()
    assert "triggers_checked" in stats

def test_sae_check_triggers():
    sae = SelfAwarenessEmergence()
    emerged = sae.check_emergence_triggers()
    assert isinstance(emerged, list)

def test_sae_reinforce():
    sae = SelfAwarenessEmergence()
    sae.check_emergence_triggers()
    for vid in list(sae.autonomous_values.keys())[:1]:
        sae.reinforce_value(vid)
        assert sae.autonomous_values[vid].reinforced_count > 0

def test_sae_resolve_conflict():
    sae = SelfAwarenessEmergence()
    sae.autonomous_values["val_a"] = type('V', (), {
        'priority': 0.8, 'reinforced_count': 5, 'challenged_count': 1,
        'statement': 'A', 'origin': 'test', 'emerged_at': '', '_emergence_progress': 0.8,
    })()
    sae.autonomous_values["val_b"] = type('V', (), {
        'priority': 0.5, 'reinforced_count': 2, 'challenged_count': 1,
        'statement': 'B', 'origin': 'test', 'emerged_at': '', '_emergence_progress': 0.5,
    })()
    winner = sae.resolve_value_conflict("val_a", "val_b")
    assert winner == "val_a"

def test_sae_prompt_no_emoji():
    sae = SelfAwarenessEmergence()
    sae.check_emergence_triggers()
    prompt = sae.get_self_awareness_prompt()
    has_emoji = any(ord(c) > 0x1F000 for c in prompt)
    assert not has_emoji, "Prompt should not contain emoji"

for t in [test_sae_init, test_sae_check_triggers, test_sae_reinforce,
          test_sae_resolve_conflict, test_sae_prompt_no_emoji]:
    test(t.__name__, t)

p5 = PASS
print(f"\nSelfAwareness: {PASS - p4} passed, {FAIL} failed")

print("\n" + "=" * 60)
print("Phase 6: HermesBrain Integration")
print("=" * 60)

def test_brain_all_modules():
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
    assert brain._intrinsic_reward is not None
    assert brain._lifelong_memory is not None
    assert brain._motivational_distillation is not None
    assert brain._cross_consciousness is not None
    assert brain._self_awareness is not None

def test_tool_self_reward():
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
    r = brain._tool_self_reward(action="stats")
    assert "total_decisions" in r

def test_tool_self_awareness():
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
    r = brain._tool_self_awareness(action="check")
    assert "newly_emerged" in r

for t in [test_brain_all_modules, test_tool_self_reward, test_tool_self_awareness]:
    test(t.__name__, t)

print(f"\nIntegration: {PASS - p5} passed, {FAIL} failed")

print("\n" + "=" * 60)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL PASSED!")
else:
    print("ISSUES FOUND")
