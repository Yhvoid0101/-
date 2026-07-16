import sys, os
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

print("=" * 60)
print("REAL SCENARIO: Complete Task Lifecycle with All Feedback Loops")
print("=" * 60)

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt, 'modules': {}})()

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

print()
print("--- SCENARIO: Hermes receives a code fix task ---")
print()

print("STEP 1: User sends task to HermesBrain")
task = "修复 agent_bus.py 中的循环依赖问题"
print(f"  Task: {task}")

print()
print("STEP 2: DeepThink analyzes the task (with causal reasoning)")
thought = dt.think(task)
print(f"  Solutions: {len(thought.proposed_solutions)}")
print(f"  Has causal summary: {'因果链推演' in thought.enhanced_prompt}")
print(f"  Risks identified: {len(thought.risks)}")
for sol in thought.proposed_solutions[:3]:
    wcs = sol.get('worst_case_scenario', 'N/A')
    print(f"    - {sol.get('name', '?')[:30]} | worst_case: {wcs[:40]}")

print()
print("STEP 3: AttentionMemory focuses on relevant dimensions")
weights = am.allocate_attention(task=task, task_type="fix")
top_dims = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
print(f"  Top attention dimensions: {[(d, f'{w:.2f}') for d, w in top_dims]}")

print()
print("STEP 4: CognitiveNetwork routes to best tool")
cn.set_input(task)
cn.propagate(iterations=5)
tool = cn.get_most_active_tool_neuron()
print(f"  Routed tool: {tool}")

print()
print("STEP 5: PredictiveEngine predicts outcome")
pred = pe.predict("fix")
print(f"  Predicted success: {pred.predicted_success:.3f}")
print(f"  Confidence: {pred.confidence:.3f}")

print()
print("STEP 6: InstinctEngine evaluates current instinct state")
ie_result = ie.evaluate(context={"task": task, "type": "fix"})
print(f"  Dominant instinct: {ie_result.instinct.value if ie_result else 'none'}")
print(f"  Action: {ie_result.name if ie_result else 'none'}")
print(f"  Score: {ie_result.weight if ie_result else 0:.3f}")

print()
print("STEP 7: Simulate task execution - MIXED RESULTS")
steps = [
    {"action": "扫描循环依赖", "success": True, "output": "发现3个循环: A->B->C->A"},
    {"action": "分析依赖图", "success": True, "output": "确认A->B->C->A是最严重的循环"},
    {"action": "移除C->A依赖", "success": True, "output": "已移除C到A的import"},
    {"action": "验证修复", "success": False, "output": "验证失败错误: C模块缺少A的引用"},
    {"action": "回滚变更", "resource": "agent_bus.py", "action_type": "write", "success": True},
    {"action": "更新agent_bus", "resource": "agent_bus.py", "action_type": "write",
     "success": False, "output": "agent_bus.py not found"},
]

for i, step in enumerate(steps):
    ti.add_raw_event(step["action"], step.get("output", ""), importance=0.7 if step["success"] else 0.9)
    if step["success"]:
        nt.record_result(True, 0.8, 'positive')
    else:
        nt.record_result(False, 0.3, 'negative')

print(f"  Steps executed: {len(steps)}")
print(f"  Success rate: {sum(1 for s in steps if s['success'])}/{len(steps)}")
print(f"  NT after execution: DA={nt.dopamine:.2f} NE={nt.norepinephrine:.2f}")

print()
print("STEP 8: MetaCognition reflects on the result")
task_result = {
    'task': task,
    'plan': {'steps': steps},
    'result': {'success': False, 'error': '验证未通过'},
    'evaluation': {'total_score': 0.6},
}
reflection = mc.reflect(task_result)
print(f"  Quality score: {reflection['quality_score']:.3f}")
print(f"  Biases detected: {len(reflection['biases_detected'])}")
for b in reflection['biases_detected'][:5]:
    print(f"    - {b}")
print(f"  Contradiction info: {len(reflection.get('contradiction_info', []))} items")
for ci in reflection.get('contradiction_info', [])[:3]:
    print(f"    - {ci['type']} ({ci['severity']}): steps {ci['steps']}")
print(f"  Suggestions: {len(reflection['improvement_suggestions'])}")
for s in reflection['improvement_suggestions'][:3]:
    print(f"    - {s}")

print()
print("STEP 9: Deep reflect with causal analysis")
deep = mc.deep_reflect(task_result)
print(f"  Depth: {deep['depth']}")
ca = deep.get('causal_analysis', {})
print(f"  Causal chain length: {ca.get('chain_length', 0)}")
print(f"  Contradictions found: {ca.get('contradictions_found', 0)}")
print(f"  High severity: {ca.get('high_severity', 0)}")
print(f"  Recommendations: {ca.get('recommendations', [])}")
for ci in deep.get('causal_interventions', []):
    print(f"  Intervention: {ci['intervention'][:60]} (conf={ci['confidence']})")

print()
print("STEP 10: Verify all feedback loops fired")
print(f"  Loop 1 (MC->DT): DT corrections = {len(dt._recent_bias_corrections)}")
print(f"  Loop 2 (PE->CN): CN synapses = {len(cn.synapses)}")
print(f"  Loop 3 (AM->TI): TI weights = {len(ti._attention_weights)}")
print(f"  Loop 4 (CT self): CT activations = {sum(ct._tool_activation.values())}")
print(f"  Loop 5 (DCR->PE): Causal boost = {pe._get_causal_boost('fix'):.4f}")
print(f"  Loop 6 (MC->IE): IE actions = {len(ie._actions)}")
print(f"  Loop 7 (MC2->CT): CT concepts = {len(ct.concepts)}")
print(f"  Loop 8 (IE->AM): AM slots = {len(am._attention_slots)}")

print()
print("STEP 11: Second task - should learn from first failure")
thought2 = dt.think("重新设计依赖注入方案以避免循环")
prompt2 = dt._build_enhanced_prompt("重新设计依赖注入方案", thought2)
has_correction = "偏差修正提醒" in prompt2
print(f"  Solutions: {len(thought2.proposed_solutions)}")
print(f"  Has correction from previous reflection: {has_correction}")
print(f"  Previous corrections in DT: {dt._recent_bias_corrections[:3]}")

pred2 = pe.predict("fix")
print(f"  Prediction adjusted: {pred2.predicted_success:.3f}")

print()
print("STEP 12: Third task after recovery - success streak")
for _ in range(5):
    nt.record_result(True, 0.9, 'positive')

thought3 = dt.think("优化模块注册机制")
print(f"  Solutions: {len(thought3.proposed_solutions)}")
print(f"  NT levels: DA={nt.dopamine:.2f} NE={nt.norepinephrine:.2f}")

mod = dcr._get_neurotransmitter_modulation()
print(f"  Causal modulation: optimism={mod['optimism']:.2f} caution={mod['caution']:.2f}")

print()
print("=" * 60)
print("REAL SCENARIO SIMULATION COMPLETE")
print("=" * 60)

all_loops_fired = (
    len(dt._recent_bias_corrections) > 0 and
    len(ti._attention_weights) > 0 and
    sum(ct._tool_activation.values()) > 0 and
    has_correction
)
print(f"All feedback loops active: {'YES' if all_loops_fired else 'NO'}")
print(f"System learned from failure: {'YES' if has_correction else 'NO'}")
print(f"Neurotransmitter modulation working: {'YES' if mod['optimism'] > 1.0 else 'NEEDS CHECK'}")
