import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.cognitive_network import CognitiveNetwork

print("=" * 60)
print("P2-1: Dynamic Neuron Spawning Verification")
print("=" * 60)

cn = CognitiveNetwork()
initial_count = len(cn.neurons)
print(f"  Initial neurons: {initial_count}")

print()
print("--- ORIGINAL PLAN TEST: Spawn after 3 appearances ---")
for i in range(3):
    cn.set_input('测试新概念 neural_plasticity')
    stats = cn.get_stats()
    print(f"  Iteration {i+1}: neurons={len(cn.neurons)}, spawned={stats['spawned_neurons']}")

print(f"  After 3 appearances: spawned={cn.get_stats()['spawned_neurons']}")
spawned_ids = cn._spawned_neuron_ids
print(f"  Spawned IDs: {spawned_ids}")

print()
print("--- EVOLVED TEST 1: Spawned neuron has correct connections ---")
if spawned_ids:
    sid = list(spawned_ids)[0]
    tool_connections = [sk for sk in cn.synapses
                        if cn.synapses[sk].source_id == sid
                        and cn.synapses[sk].target_id.startswith("tool_")]
    assoc_connections = [sk for sk in cn.synapses
                          if cn.synapses[sk].source_id == sid
                          and cn.synapses[sk].target_id.startswith("assoc_")]
    mem_connections = [sk for sk in cn.synapses
                       if cn.synapses[sk].source_id == sid
                       and cn.synapses[sk].target_id.startswith("mem_")]
    print(f"  Tool connections: {len(tool_connections)}")
    print(f"  Association connections: {len(assoc_connections)}")
    print(f"  Memory connections: {len(mem_connections)}")
    print(f"  Threshold: {cn.neurons[sid].threshold}")
    print(f"  Label: {cn.neurons[sid].label}")

print()
print("--- EVOLVED TEST 2: Spawned neuron is matchable ---")
cn2 = CognitiveNetwork()
for _ in range(3):
    cn2.set_input('quantum_computing task')
print(f"  Spawned after 3: {cn2.get_stats()['spawned_neurons']}")

cn2.set_input('quantum_computing analysis')
has_sensor = any('quantum_computing' in k for k in cn2.sensor_neurons)
print(f"  quantum_computing in sensor_neurons: {has_sensor}")

cn2.propagate(iterations=3)
tool = cn2.get_most_active_tool_neuron()
print(f"  Routed tool for 'quantum_computing analysis': {tool}")

print()
print("--- EVOLVED TEST 3: Multiple concepts spawning ---")
cn3 = CognitiveNetwork()
concepts = ['blockchain', 'metaverse', 'kubernetes', 'terraform', 'graphql']
for concept in concepts:
    for _ in range(3):
        cn3.set_input(f'task about {concept}')
    spawned = cn3.get_stats()['spawned_neurons']
    print(f"  After '{concept}': spawned={spawned}")

print(f"  Total spawned: {cn3.get_stats()['spawned_neurons']}")

print()
print("--- EVOLVED TEST 4: Max spawned limit ---")
cn4 = CognitiveNetwork()
cn4._max_spawned_neurons = 5
for i in range(10):
    for _ in range(3):
        cn4.set_input(f'concept_alpha_{i}')
    stats = cn4.get_stats()
    if stats['spawned_neurons'] >= 5:
        print(f"  Hit limit at concept {i}: spawned={stats['spawned_neurons']}")
        break

print(f"  Final spawned: {cn4.get_stats()['spawned_neurons']} (max=5)")

print()
print("--- EVOLVED TEST 5: Chinese concept spawning ---")
cn5 = CognitiveNetwork()
for _ in range(3):
    cn5.set_input('深度学习模型训练')
print(f"  Chinese spawn: {cn5.get_stats()['spawned_neurons']}")
if cn5._spawned_neuron_ids:
    for sid in cn5._spawned_neuron_ids:
        print(f"    {cn5.neurons[sid].label}")

print()
print("--- EVOLVED TEST 6: Existing concepts NOT re-spawned ---")
cn6 = CognitiveNetwork()
for _ in range(5):
    cn6.set_input('修复代码bug')
spawned = cn6.get_stats()['spawned_neurons']
print(f"  Existing concept '修复' spawn count: {spawned} (should be 0)")

print()
print("--- EVOLVED TEST 7: Concurrent spawning ---")
errors = []
cn7 = CognitiveNetwork()

def worker(tid, iters):
    for i in range(iters):
        try:
            cn7.set_input(f'concurrent_concept_{tid}_{i % 3}')
        except Exception as e:
            errors.append(f"T{tid}-{i}: {e}")

threads = [threading.Thread(target=worker, args=(t, 10)) for t in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"  Concurrent errors: {len(errors)}")
print(f"  Spawned after concurrent: {cn7.get_stats()['spawned_neurons']}")

print()
print("--- EVOLVED TEST 8: Stats include spawned info ---")
stats = cn3.get_stats()
print(f"  spawned_neurons: {stats.get('spawned_neurons', 'N/A')}")
print(f"  concept_activation_pending: {len(stats.get('concept_activation_pending', {}))}")

print()
print("--- EVOLVED TEST 9: Regression - existing routing still works ---")
cn9 = CognitiveNetwork()
cn9.set_input('修复 agent_bus.py 的长函数')
cn9.propagate(iterations=5)
tool = cn9.get_most_active_tool_neuron()
print(f"  Route '修复...': {tool}")

cn9.set_input('检查系统安全漏洞')
cn9.propagate(iterations=5)
tool2 = cn9.get_most_active_tool_neuron()
print(f"  Route '安全...': {tool2}")

print()
print("ALL P2-1 TESTS COMPLETED!")
