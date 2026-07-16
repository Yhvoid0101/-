import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

modules = [
    'core.hermes_brain', 'core.deep_think', 'core.cognitive_network',
    'core.attention_memory', 'core.instinct_engine', 'core.meta_cognition',
    'core.predictive_engine', 'core.temporal_integration',
    'core.cognitive_topology', 'core.deep_causal_reasoner',
    'core.neurotransmitter_system', 'core.memory_crystallizer',
    'core.obsidian_bridge', 'core.adaptive_router',
]

print("REGRESSION: All Module Imports After Feedback Loop Changes")
print("=" * 60)
passed = 0
for m in modules:
    try:
        __import__(m)
        print(f'  OK: {m}')
        passed += 1
    except Exception as e:
        print(f'  FAIL: {m}: {e}')

print(f"\nResult: {passed}/{len(modules)}")

from core.deep_think import DeepThink
from core.meta_cognition import MetaCognitionEngine as MetaCognition
from core.neurotransmitter_system import NeuroTransmitterSystem
from core.attention_memory import AttentionMemory
from core.instinct_engine import InstinctEngine, InstinctType
from core.predictive_engine import PredictiveEngine
from core.cognitive_topology import CognitiveTopology
from core.temporal_integration import TemporalIntegration
from core.deep_causal_reasoner import DeepCausalReasoner

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt})()

dt = DeepThink(bus=bus)
mc = MetaCognition(bus=bus)
am = AttentionMemory(bus=bus)
ie = InstinctEngine(bus=bus)
pe = PredictiveEngine(bus=bus)
ct = CognitiveTopology(bus=bus)
ti = TemporalIntegration(bus=bus)
dcr = DeepCausalReasoner(bus=bus)

bus.deep_think = dt
bus.meta_cognition = mc
bus.attention_memory = am
bus.instinct_engine = ie
bus.predictive_engine = pe
bus.cognitive_topology = ct
bus.temporal_integration = ti
bus.deep_causal_reasoner = dcr

print("\nFUNCTIONAL CHECKS:")
checks = 0

dt.receive_bias_corrections(["test correction"])
checks += 1; print(f"  DeepThink.receive_bias_corrections: OK")

ti.receive_attention_weights({"risk": 0.8})
checks += 1; print(f"  TemporalIntegration.receive_attention_weights: OK")

ie.modulate_cluster("explore", 1.1)
checks += 1; print(f"  InstinctEngine.modulate_cluster: OK")

am.apply_instinct_influence({"risk": 1.3})
checks += 1; print(f"  AttentionMemory.apply_instinct_influence: OK")

ct.record_tool_usage("analyze_code")
checks += 1; print(f"  CognitiveTopology.record_tool_usage: OK")

boost = pe._get_causal_boost("fix")
checks += 1; print(f"  PredictiveEngine._get_causal_boost: {boost:.4f}")

thought = dt.think("test task")
checks += 1; print(f"  DeepThink.think: {len(thought.proposed_solutions)} solutions")

reflection = mc.reflect({'task': 'test', 'plan': {'steps': [{'action': 'a'}]},
    'result': {'success': True}, 'evaluation': {'total_score': 0.9}})
checks += 1; print(f"  MetaCognition.reflect: quality={reflection['quality_score']:.3f}")

print(f"\nAll {checks} functional checks passed!")
print(f"\nTOTAL: {passed} imports + {checks} functions = {passed + checks} checks")
