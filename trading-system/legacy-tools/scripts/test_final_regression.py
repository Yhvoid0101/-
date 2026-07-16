import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.deep_causal_reasoner import DeepCausalReasoner, CausalGraph, CausalNode
from core.deep_think import DeepThink
from core.meta_cognition import MetaCognition
from core.neurotransmitter_system import NeuroTransmitterSystem

print("FINAL REGRESSION TEST")
print("=" * 60)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f'  PASS: {name}')
    else:
        failed += 1
        print(f'  FAIL: {name}')

dcr = DeepCausalReasoner()

chain = dcr.trace_causal_chain('test', [{'detail': 'a'}, {'detail': 'b'}])
check('Chain tracing', len(chain) == 3)

cf = dcr.counterfactual_reasoning(chain, 'alt')
check('Counterfactual', 'hypothetical_chain' in cf)

contra = dcr.detect_contradiction(
    {'action': 'x', 'success': True, 'output': 'ok'},
    {'action': 'y', 'success': False, 'output': 'not found'})
check('Contradiction detection', contra is not None)

batch = dcr.detect_contradictions_batch([
    {'action': 'a', 'success': True, 'output': 'ok'},
    {'action': 'b', 'success': False, 'output': 'missing'},
])
check('Batch contradiction', len(batch) >= 1)

validation = dcr.validate_chain(chain)
check('Chain validation', 'valid' in validation)

merged = dcr.merge_chains(chain, chain)
check('Chain merge', len(merged) >= len(chain))

summary = dcr.generate_causal_summary('test', chain)
check('Causal summary', '因果链推演' in summary)

stats = dcr.get_stats()
check('Stats', 'chains_traced' in stats)

graph = dcr.causal_graph
check('Causal graph exists', graph.size() > 0)
check('Graph no cycle', not graph.has_cycle())

intervention = dcr.intervene(chain, chain[1].id, 'new')
check('Intervention', 'intervention' in intervention)

sim = dcr._compute_similarity('hello', 'hello')
check('Similarity identical', sim == 1.0)
sim_diff = dcr._compute_similarity('abc', 'xyz')
check('Similarity different', sim_diff < 0.3)

nt = NeuroTransmitterSystem()
bus = type('Bus', (), {'neurotransmitter': nt})()

dt = DeepThink(bus=bus)
check('DeepThink init', dt._causal_reasoner is not None)

thought = dt.think('test task')
check('DeepThink think', len(thought.proposed_solutions) > 0)
check('DeepThink causal in prompt', '因果链推演' in thought.enhanced_prompt)

mc = MetaCognition(bus=bus)
check('MetaCognition init', mc._causal_reasoner is not None)

reflection = mc.reflect({
    'task': 'test', 'plan': {'steps': [{'action': 'a'}]},
    'result': {'success': True}, 'evaluation': {'total_score': 0.9}
})
check('MetaCognition reflect', 'quality_score' in reflection)
check('MetaCognition contradiction_info', 'contradiction_info' in reflection)

deep = mc.deep_reflect({
    'task': 'test', 'plan': {'steps': [{'action': 'a'}, {'action': 'b'}]},
    'result': {'success': False}, 'evaluation': {'total_score': 0.5}
})
check('Deep reflect', deep['depth'] == 'deep')
check('Deep reflect causal_analysis', 'causal_analysis' in deep)
check('Deep reflect causal_interventions', 'causal_interventions' in deep)

nt.record_result(True, 0.9, 'positive')
mod = dcr._get_neurotransmitter_modulation()
check('Neurotransmitter modulation', 'optimism' in mod)

dcr_edge = DeepCausalReasoner()
empty_val = dcr_edge.validate_chain([])
check('Empty chain validation', not empty_val['valid'])

empty_cf = dcr_edge.counterfactual_reasoning([], 'alt')
check('Empty chain counterfactual', 'hypothetical' in empty_cf)

graph_dag = CausalGraph()
n1 = CausalNode(event='A', id='x')
n2 = CausalNode(event='B', id='y')
graph_dag.add_node(n1)
graph_dag.add_node(n2)
graph_dag.add_edge('x', 'y')
check('DAG children', len(graph_dag.get_children('x')) == 1)
check('DAG parents', len(graph_dag.get_parents('y')) == 1)
check('DAG roots', len(graph_dag.get_roots()) == 1)
check('DAG leaves', len(graph_dag.get_leaves()) == 1)

paths = graph_dag.find_all_paths('x', 'y')
check('DAG path finding', len(paths) == 1)

topo = graph_dag.topological_sort()
check('DAG topological sort', len(topo) == 2)

print()
print(f'RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests')
if failed == 0:
    print('ALL TESTS PASSED!')
else:
    print(f'WARNING: {failed} tests failed!')
