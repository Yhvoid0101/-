import sys
import os
import threading
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.deep_causal_reasoner import DeepCausalReasoner, CausalGraph, CausalNode
from core.neurotransmitter_system import NeuroTransmitterSystem

print("=" * 60)
print("DEEP TEST 1: Concurrent Access Safety")
print("=" * 60)

dcr = DeepCausalReasoner()
errors = []

def worker(thread_id):
    try:
        for i in range(10):
            events = [{'detail': f'线程{thread_id}事件{j}'} for j in range(3)]
            chain = dcr.trace_causal_chain(f'线程{thread_id}任务{i}', events)
            dcr.validate_chain(chain)
            dcr.counterfactual_reasoning(chain, f'替代方案{thread_id}_{i}')
    except Exception as e:
        errors.append(f'Thread {thread_id}: {e}')

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f'  Threads: 5, Errors: {len(errors)}')
if errors:
    for e in errors[:3]:
        print(f'  ERROR: {e}')
else:
    print(f'  All threads completed safely')
    stats = dcr.get_stats()
    print(f'  Chains traced: {stats["chains_traced"]}')
    print(f'  Counterfactuals: {stats["counterfactuals_run"]}')

print()
print("=" * 60)
print("DEEP TEST 2: Long Chain + Pruning")
print("=" * 60)

dcr2 = DeepCausalReasoner()
long_events = [{'detail': f'步骤{i+1}: 执行操作{i+1}', 'success': True} for i in range(20)]
long_chain = dcr2.trace_causal_chain('超长任务', long_events)
print(f'  Chain length (20 events): {len(long_chain)}')

validation = dcr2.validate_chain(long_chain)
print(f'  Valid: {validation["valid"]}')
print(f'  Issues: {len(validation["issues"])}')
for issue in validation["issues"][:5]:
    print(f'    {issue["type"]}: pos={issue.get("position", "?")}')

print()
print("=" * 60)
print("DEEP TEST 3: Weak Link Detection + Pruning")
print("=" * 60)

dcr3 = DeepCausalReasoner()
dcr3.WEAK_LINK_THRESHOLD = 0.4

weak_events = [
    {'detail': '强因果步骤A'},
    {'detail': '弱因果步骤B'},
    {'detail': '强因果步骤C'},
]
weak_chain = dcr3.trace_causal_chain('弱连接测试', weak_events)

if len(weak_chain) > 1:
    weak_chain[1].confidence = 0.15
    weak_chain[1].strength = 0.1

validation = dcr3.validate_chain(weak_chain)
print(f'  Valid: {validation["valid"]}')
print(f'  Pruned indices: {validation["pruned_indices"]}')
print(f'  Original length: {validation["original_length"]}')
print(f'  Pruned length: {validation["pruned_length"]}')

print()
print("=" * 60)
print("DEEP TEST 4: Duplicate Event Detection")
print("=" * 60)

dcr4 = DeepCausalReasoner()
dup_events = [
    {'detail': '扫描代码'},
    {'detail': '修复bug'},
    {'detail': '扫描代码'},
    {'detail': '测试验证'},
]
dup_chain = dcr4.trace_causal_chain('重复步骤测试', dup_events)
validation = dcr4.validate_chain(dup_chain)
print(f'  Valid: {validation["valid"]}')
dup_issues = [i for i in validation["issues"] if i["type"] == "duplicate_event"]
print(f'  Duplicate events detected: {len(dup_issues)}')
for di in dup_issues:
    print(f'    Position {di["position"]}: {di["event"]}')

print()
print("=" * 60)
print("DEEP TEST 5: Confidence Decay Over Time")
print("=" * 60)

dcr5 = DeepCausalReasoner()
chain5 = dcr5.trace_causal_chain('衰减测试', [{'detail': '步骤1'}, {'detail': '步骤2'}])
original_conf = [n.confidence for n in chain5]

dcr5._chain_timestamps[0] = time.time() - 7200
dcr5.apply_confidence_decay()
decayed_conf = [n.confidence for n in chain5]

print(f'  Original confidence: {original_conf}')
print(f'  After 2h decay: {decayed_conf}')
print(f'  Decay observed: {original_conf != decayed_conf}')

print()
print("=" * 60)
print("DEEP TEST 6: All 8 Contradiction Patterns")
print("=" * 60)

dcr6 = DeepCausalReasoner()

test_pairs = [
    ("dependency_failure", {'action': 'A', 'success': True, 'output': 'ok'}, {'action': 'B', 'success': False, 'output': 'not found'}),
    ("conflicting_modification", {'action': 'A', 'success': True, 'modified': True}, {'action': 'B', 'success': False, 'modified': True}),
    ("expectation_mismatch", {'action': 'A', 'expected_output': '成功通过完成'}, {'action': 'B', 'output': '失败错误异常'}),
    ("resource_conflict", {'action': 'A', 'resource': 'file.py', 'action_type': 'write'}, {'action': 'B', 'resource': 'file.py', 'action_type': 'write'}),
    ("semantic_contradiction", {'action': 'A', 'conclusion': 'yes', 'topic': 'X'}, {'action': 'B', 'conclusion': 'no', 'topic': 'X'}),
    ("output_input_mismatch", {'action': 'A', 'success': True, 'output': 'result_x'}, {'action': 'B', 'requires_input': 'result_y'}),
    ("duplicate_effort", {'action': 'scan', 'resource': 'code.py'}, {'action': 'scan', 'resource': 'code.py'}),
]

detected_types = []
for name, a, b in test_pairs:
    result = dcr6.detect_contradiction(a, b)
    if result:
        types = [c["type"] for c in result["contradictions"]]
        detected_types.extend(types)
        status = "✓" if name in types else f"→{types[0]}"
    else:
        status = "✗ NOT DETECTED"
    print(f'  {name}: {status}')

print(f'  Total patterns detected: {len(set(detected_types))}/8')

print()
print("=" * 60)
print("DEEP TEST 7: CausalGraph Cycle Detection")
print("=" * 60)

graph = CausalGraph()
na = CausalNode(event="A", id="ca")
nb = CausalNode(event="B", id="cb")
nc = CausalNode(event="C", id="cc")
graph.add_node(na)
graph.add_node(nb)
graph.add_node(nc)
graph.add_edge("ca", "cb")
graph.add_edge("cb", "cc")
print(f'  DAG (A→B→C) has cycle: {graph.has_cycle()}')

graph.add_edge("cc", "ca")
print(f'  After adding C→A: has cycle: {graph.has_cycle()}')

print()
print("=" * 60)
print("DEEP TEST 8: Neurotransmitter Extreme Values")
print("=" * 60)

nt_extreme = NeuroTransmitterSystem()
for _ in range(20):
    nt_extreme.record_result(True, 1.0, 'positive')

bus_extreme = type('Bus', (), {'neurotransmitter': nt_extreme})()
dcr_extreme = DeepCausalReasoner(bus=bus_extreme)

mod = dcr_extreme._get_neurotransmitter_modulation()
print(f'  Extreme positive modulation: {mod}')

chain_extreme = dcr_extreme.trace_causal_chain('极端乐观', [{'detail': '步骤1'}, {'detail': '步骤2'}])
print(f'  Extreme chain confidence: {[f"{n.confidence:.3f}" for n in chain_extreme]}')

nt_extreme2 = NeuroTransmitterSystem()
for _ in range(20):
    nt_extreme2.record_result(False, 0.0, 'negative')

bus_extreme2 = type('Bus', (), {'neurotransmitter': nt_extreme2})()
dcr_extreme2 = DeepCausalReasoner(bus=bus_extreme2)

mod2 = dcr_extreme2._get_neurotransmitter_modulation()
print(f'  Extreme negative modulation: {mod2}')

chain_extreme2 = dcr_extreme2.trace_causal_chain('极端悲观', [{'detail': '步骤1'}, {'detail': '步骤2'}])
print(f'  Extreme chain confidence: {[f"{n.confidence:.3f}" for n in chain_extreme2]}')

print()
print("=" * 60)
print("DEEP TEST 9: Similarity Edge Cases")
print("=" * 60)

dcr_sim = DeepCausalReasoner()
tests = [
    ("", ""),
    ("a", ""),
    ("", "b"),
    ("相同文本", "相同文本"),
    ("完全不同", "毫无关联"),
    ("修复代码bug", "修复代码缺陷"),
    ("refactor code", "refactor code"),
    ("部署服务器", "部署应用"),
]
for a, b in tests:
    sim = dcr_sim._compute_similarity(a, b)
    print(f'  sim("{a[:20]}", "{b[:20]}") = {sim}')

print()
print("=" * 60)
print("DEEP TEST 10: Chain Merge with Overlap")
print("=" * 60)

dcr_merge = DeepCausalReasoner()
chain_a = dcr_merge.trace_causal_chain('任务A', [
    {'detail': '共享步骤1'},
    {'detail': 'A特有步骤'},
])
chain_b = dcr_merge.trace_causal_chain('任务B', [
    {'detail': '共享步骤1'},
    {'detail': 'B特有步骤'},
])

merged = dcr_merge.merge_chains(chain_a, chain_b)
events_in_merged = [n.event[:30] for n in merged]
print(f'  Chain A: {len(chain_a)} steps')
print(f'  Chain B: {len(chain_b)} steps')
print(f'  Merged: {len(merged)} steps (deduped)')
shared_count = sum(1 for e in events_in_merged if '共享' in e)
print(f'  Shared events in merged: {shared_count}')

print()
print("ALL DEEP TESTS COMPLETED!")
