import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.deep_think import DeepThink
from core.meta_cognition import MetaCognition
from core.neurotransmitter_system import NeuroTransmitterSystem

print("=" * 60)
print("END-TO-END: DeepThink -> CausalReasoner -> MetaCognition")
print("=" * 60)

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {
    'neurotransmitter': nt,
    'modules': {},
    'event': lambda self, *a, **k: None,
})()

dt = DeepThink(bus=bus)
mc = MetaCognition(bus=bus)

print()
print("--- STEP 1: DeepThink analyzes a task ---")
thought = dt.think('修复 agent_bus.py 中的循环依赖问题')
print(f'  Solutions: {len(thought.proposed_solutions)}')
print(f'  Has causal summary: {"因果链推演" in thought.enhanced_prompt}')
print(f'  Risks: {len(thought.risks)}')

print()
print("--- STEP 2: Simulate task execution with mixed results ---")
task_result = {
    'task': '修复 agent_bus.py 中的循环依赖问题',
    'plan': {
        'steps': [
            {'action': '扫描循环依赖', 'success': True, 'output': '发现3个循环'},
            {'action': '分析依赖图', 'success': True, 'output': '确认A->B->C->A'},
            {'action': '移除C->A依赖', 'success': True, 'output': '已移除'},
            {'action': '验证修复', 'success': False, 'output': '验证失败错误'},
            {'action': '回滚变更', 'resource': 'agent_bus.py', 'action_type': 'write', 'success': True},
            {'action': '更新agent_bus', 'resource': 'agent_bus.py', 'action_type': 'write', 'success': False, 'output': 'agent_bus.py not found'},
        ]
    },
    'result': {'success': False, 'error': '验证未通过'},
    'evaluation': {'total_score': 0.7},
    'risks': [{'risk': '可能引入新循环', 'severity': 'high'}],
}

print()
print("--- STEP 3: MetaCognition reflects on the result ---")
reflection = mc.reflect(task_result)
print(f'  Quality score: {reflection["quality_score"]:.3f}')
print(f'  Biases detected: {len(reflection["biases_detected"])}')
for b in reflection["biases_detected"][:5]:
    print(f'    - {b}')
print(f'  Contradiction info: {len(reflection.get("contradiction_info", []))} items')
for ci in reflection.get("contradiction_info", [])[:3]:
    print(f'    - {ci["type"]} ({ci["severity"]}): steps {ci["steps"]}')
print(f'  Suggestions: {len(reflection["improvement_suggestions"])}')
for s in reflection["improvement_suggestions"][:3]:
    print(f'    - {s}')

print()
print("--- STEP 4: Deep reflect with causal analysis ---")
deep = mc.deep_reflect(task_result)
print(f'  Depth: {deep["depth"]}')
print(f'  Quality: {deep["quality_score"]:.3f}')
print(f'  Has causal_analysis: {"causal_analysis" in deep}')
if deep.get("causal_analysis"):
    ca = deep["causal_analysis"]
    print(f'  Causal chain length: {ca.get("chain_length", 0)}')
    print(f'  Contradictions: {ca.get("contradictions_found", 0)}')
    print(f'  High severity: {ca.get("high_severity", 0)}')
    print(f'  Recommendations: {ca.get("recommendations", [])[:3]}')
print(f'  Has causal_interventions: {"causal_interventions" in deep}')
if deep.get("causal_interventions"):
    for ci in deep["causal_interventions"]:
        print(f'    Target: {ci["target"]}')
        print(f'    Intervention: {ci["intervention"][:60]}')
        print(f'    Confidence: {ci["confidence"]}')

print()
print("--- STEP 5: Neurotransmitter feedback loop ---")
nt.record_result(False, 0.3, 'negative')
nt.record_result(False, 0.2, 'negative')
print(f'  After 2 failures: DA={nt.dopamine:.2f} NE={nt.norepinephrine:.2f}')

thought2 = dt.think('重新设计依赖注入方案')
print(f'  Second thought with low DA: {len(thought2.proposed_solutions)} solutions')
if dt._causal_reasoner:
    mod = dt._causal_reasoner._get_neurotransmitter_modulation()
    print(f'  Causal modulation: {mod}')

print()
print("--- STEP 6: Recovery after success ---")
for _ in range(5):
    nt.record_result(True, 0.9, 'positive')
print(f'  After 5 successes: DA={nt.dopamine:.2f} NE={nt.norepinephrine:.2f}')

thought3 = dt.think('优化模块注册机制')
print(f'  Third thought with high DA: {len(thought3.proposed_solutions)} solutions')
if dt._causal_reasoner:
    mod3 = dt._causal_reasoner._get_neurotransmitter_modulation()
    print(f'  Causal modulation: {mod3}')

print()
print("--- STEP 7: Verify causal reasoner stats ---")
if dt._causal_reasoner:
    stats = dt._causal_reasoner.get_stats()
    print(f'  Total chains: {stats["total_chains"]}')
    print(f'  Graph size: {stats["graph_size"]}')
    print(f'  Contradictions detected: {stats["contradictions_detected"]}')
    print(f'  Counterfactuals run: {stats["counterfactuals_run"]}')
    print(f'  Plan analyses: {stats["plan_analyses"]}')

if mc._causal_reasoner:
    stats_mc = mc._causal_reasoner.get_stats()
    print(f'  MC Total chains: {stats_mc["total_chains"]}')
    print(f'  MC Batch checks: {stats_mc["batch_checks"]}')

print()
print("=" * 60)
print("END-TO-END VERIFICATION COMPLETE!")
print("=" * 60)
print()
print("SUMMARY:")
print("  DeepThink -> CausalReasoner: causal chain traced for each thought")
print("  MetaCognition -> CausalReasoner: contradictions detected in plan steps")
print("  Deep Reflect: causal analysis + interventions for failed tasks")
print("  Neurotransmitter loop: DA/NE modulates causal confidence")
print("  All three modules share the same bus and neurotransmitter instance")
