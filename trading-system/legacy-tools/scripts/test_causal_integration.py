import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("INTEGRATION TEST 1: DeepThink + CausalReasoner")
print("=" * 60)

from core.deep_think import DeepThink
from core.neurotransmitter_system import NeuroTransmitterSystem

nt = NeuroTransmitterSystem()
for _ in range(5):
    nt.record_result(True, 0.9, 'positive')

bus = type('Bus', (), {'neurotransmitter': nt})()
dt = DeepThink(bus=bus)

print(f'  CausalReasoner initialized: {dt._causal_reasoner is not None}')
print(f'  AttentionMemory initialized: {dt._attention_memory is not None}')
print(f'  NeuroTransmitter connected: {dt._neurotransmitter is not None}')

result = dt.think('重构 agent_bus.py 的事件路由逻辑')
print(f'  Solutions: {len(result.proposed_solutions)}')
print(f'  Risks: {len(result.risks)}')
print(f'  Ambiguities: {len(result.identified_ambiguities)}')
print(f'  Has causal summary in prompt: {"因果链推演" in result.enhanced_prompt}')
print(f'  Duration: {result.duration_ms:.1f}ms')

if result.proposed_solutions:
    for i, sol in enumerate(result.proposed_solutions):
        wcs = sol.get('worst_case_scenario', 'N/A')
        print(f'  Solution {i}: {sol.get("name", "?")[:30]} | worst_case: {wcs[:50]}')

if dt._causal_reasoner:
    stats = dt._causal_reasoner.get_stats()
    print(f'  Causal chains traced: {stats["chains_traced"]}')
    print(f'  Graph size: {stats["graph_size"]}')

print()
print("=" * 60)
print("INTEGRATION TEST 2: MetaCognition + CausalReasoner")
print("=" * 60)

from core.meta_cognition import MetaCognition

mc = MetaCognition(bus=bus)
print(f'  CausalReasoner initialized: {mc._causal_reasoner is not None}')

result_a = {
    'task': '修复 agent_bus.py 的长函数',
    'plan': {'steps': [
        {'action': '使用代码格式化工具', 'success': True, 'output': '格式化完成'},
    ]},
    'result': {'success': True},
    'evaluation': {'total_score': 0.9}
}
reflection_a = mc.reflect(result_a)
print(f'  Scenario A (single plan): quality={reflection_a["quality_score"]:.3f}')
print(f'  Biases: {reflection_a["biases_detected"][:3]}')
print(f'  Has contradiction_info: {"contradiction_info" in reflection_a}')

result_b = {
    'task': '优化所有模块的导入',
    'plan': {'steps': [
        {'action': '方案A', 'success': True, 'output': 'found bug in agent_bus.py'},
        {'action': '方案B', 'success': False, 'output': 'agent_bus.py not found'},
        {'action': '方案C', 'success': False, 'output': '失败错误异常', 'expected_output': '成功通过'},
    ]},
    'result': {'success': False, 'error': '引入循环依赖'},
    'evaluation': {'total_score': 0.95}
}
reflection_b = mc.reflect(result_b)
print(f'  Scenario B (overconfident): quality={reflection_b["quality_score"]:.3f}')
print(f'  Biases: {reflection_b["biases_detected"][:5]}')
print(f'  Contradiction info: {len(reflection_b.get("contradiction_info", []))} items')
for ci in reflection_b.get("contradiction_info", [])[:3]:
    print(f'    {ci["type"]} ({ci["severity"]}): steps {ci["steps"]}')

print()
print("=" * 60)
print("INTEGRATION TEST 3: Deep Reflect with Causal Analysis")
print("=" * 60)

result_c = {
    'task': '部署新版本到生产环境',
    'plan': {'steps': [
        {'action': '运行测试', 'success': True, 'output': '测试全部通过'},
        {'action': '构建镜像', 'success': True, 'output': '镜像构建成功'},
        {'action': '部署服务', 'success': False, 'output': '部署失败错误'},
        {'action': '回滚', 'resource': 'config.yaml', 'action_type': 'write', 'success': True},
        {'action': '更新配置', 'resource': 'config.yaml', 'action_type': 'write', 'success': False},
    ]},
    'result': {'success': False, 'error': '部署超时'},
    'evaluation': {'total_score': 0.6}
}
deep = mc.deep_reflect(result_c)
print(f'  Depth: {deep["depth"]}')
print(f'  Quality: {deep["quality_score"]:.3f}')
print(f'  Has causal_analysis: {"causal_analysis" in deep}')
print(f'  Has causal_interventions: {"causal_interventions" in deep}')

if deep.get("causal_analysis"):
    ca = deep["causal_analysis"]
    print(f'  Causal chain length: {ca.get("chain_length", 0)}')
    print(f'  Contradictions found: {ca.get("contradictions_found", 0)}')
    print(f'  High severity: {ca.get("high_severity", 0)}')
    print(f'  Chain valid: {ca.get("chain_valid", True)}')
    print(f'  Causal recommendations: {ca.get("recommendations", [])[:3]}')

if deep.get("causal_interventions"):
    for ci in deep["causal_interventions"]:
        print(f'  Intervention: {ci["intervention"][:60]} (conf={ci["confidence"]})')

print()
print("=" * 60)
print("INTEGRATION TEST 4: Neurotransmitter Modulation")
print("=" * 60)

from core.deep_causal_reasoner import DeepCausalReasoner

nt_low = NeuroTransmitterSystem()
for _ in range(5):
    nt_low.record_result(False, 0.2, 'negative')

bus_low = type('Bus', (), {'neurotransmitter': nt_low})()
dcr_low = DeepCausalReasoner(bus=bus_low)

chain_high = DeepCausalReasoner(bus=type('Bus', (), {'neurotransmitter': nt})()).trace_causal_chain(
    '高多巴胺任务', [{'detail': '步骤1'}, {'detail': '步骤2'}])
chain_low = dcr_low.trace_causal_chain(
    '低多巴胺任务', [{'detail': '步骤1'}, {'detail': '步骤2'}])

print(f'  High dopamine chain conf: {[f"{n.confidence:.3f}" for n in chain_high]}')
print(f'  Low dopamine chain conf: {[f"{n.confidence:.3f}" for n in chain_low]}')

mod_high = DeepCausalReasoner(bus=type('Bus', (), {'neurotransmitter': nt})())._get_neurotransmitter_modulation()
mod_low = dcr_low._get_neurotransmitter_modulation()
print(f'  High DA modulation: {mod_high}')
print(f'  Low DA modulation: {mod_low}')

print()
print("=" * 60)
print("INTEGRATION TEST 5: Edge Cases")
print("=" * 60)

dcr_edge = DeepCausalReasoner()

empty_chain = dcr_edge.trace_causal_chain('空事件', [])
print(f'  Empty events chain: {len(empty_chain)} steps')

cf_empty = dcr_edge.counterfactual_reasoning([], '替代方案')
print(f'  Empty chain counterfactual: {cf_empty.get("hypothetical", "N/A")[:50]}')

validation_empty = dcr_edge.validate_chain([])
print(f'  Empty chain validation: valid={validation_empty["valid"]}')

no_contra = dcr_edge.detect_contradiction({}, {})
print(f'  Empty steps contradiction: {no_contra}')

batch_empty = dcr_edge.detect_contradictions_batch([])
print(f'  Empty batch: {len(batch_empty)} contradictions')

batch_single = dcr_edge.detect_contradictions_batch([{'action': 'test'}])
print(f'  Single step batch: {len(batch_single)} contradictions')

merged_empty = dcr_edge.merge_chains([], [])
print(f'  Merge two empty chains: {len(merged_empty)} steps')

intervene_not_found = dcr_edge.intervene([], 'nonexistent', 'new')
print(f'  Intervene on empty chain: {intervene_not_found["changed"]}')

print()
print("=" * 60)
print("INTEGRATION TEST 6: CausalGraph Advanced")
print("=" * 60)

from core.deep_causal_reasoner import CausalGraph, CausalNode

graph = CausalGraph()
n1 = CausalNode(event="A", id="a")
n2 = CausalNode(event="B", id="b")
n3 = CausalNode(event="C", id="c")
n4 = CausalNode(event="D", id="d")

graph.add_node(n1)
graph.add_node(n2)
graph.add_node(n3)
graph.add_node(n4)
graph.add_edge("a", "b")
graph.add_edge("a", "c")
graph.add_edge("b", "d")
graph.add_edge("c", "d")

print(f'  DAG size: {graph.size()}')
print(f'  Has cycle: {graph.has_cycle()}')
print(f'  Roots: {[n.event for n in graph.get_roots()]}')
print(f'  Leaves: {[n.event for n in graph.get_leaves()]}')
print(f'  Children of A: {[n.event for n in graph.get_children("a")]}')
print(f'  Parents of D: {[n.event for n in graph.get_parents("d")]}')

paths = graph.find_all_paths("a", "d")
print(f'  Paths A→D: {len(paths)}')
for p in paths:
    print(f'    {" → ".join(p)}')

topo = graph.topological_sort()
print(f'  Topological order: {topo}')

graph_dict = graph.to_dict()
print(f'  Graph dict keys: {list(graph_dict.keys())}')

print()
print("ALL INTEGRATION TESTS COMPLETED!")
