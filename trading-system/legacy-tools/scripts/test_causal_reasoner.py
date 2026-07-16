import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.deep_causal_reasoner import DeepCausalReasoner, CausalGraph, CausalNode

dcr = DeepCausalReasoner()

print("=" * 60)
print("TEST 1: Causal Chain Tracing")
print("=" * 60)
events = [
    {'detail': 'pylint扫描发现3个长函数'},
    {'detail': '分析确认需要拆分process_event'},
    {'detail': '执行拆分，将process_event分为4个子函数'},
    {'detail': '运行测试，验证拆分后功能正常'},
]
chain = dcr.trace_causal_chain('启动代码修复', events)
print(f'  Chain length: {len(chain)}')
print(f'  Root: {chain[0].event}')
print(f'  Final effect: {chain[-1].effects[0] if chain[-1].effects else chain[-1].event}')
print(f'  Confidence range: {chain[-1].confidence:.3f} - {chain[0].confidence:.3f}')
for i, node in enumerate(chain):
    print(f'  Step {i}: {node.event[:50]} [conf={node.confidence:.3f}]')

print()
print("=" * 60)
print("TEST 2: Counterfactual Reasoning")
print("=" * 60)
cf = dcr.counterfactual_reasoning(chain, '手动重构而非自动拆分')
print(f'  Original: {cf["original_action"]}')
print(f'  Alternative: {cf["alternative_action"]}')
print(f'  Hypothetical chain: {cf["hypothetical_chain"][:3]}')
print(f'  Confidence: {cf["confidence"]}')
print(f'  Similarity: {cf["similarity"]}')

print()
print("=" * 60)
print("TEST 3: Contradiction Detection")
print("=" * 60)
step_a = {'action': 'pylint', 'output': 'found bug in agent_bus.py', 'success': True}
step_b = {'action': 'auto_fix', 'output': 'agent_bus.py not found', 'success': False}
contra = dcr.detect_contradiction(step_a, step_b)
print(f'  Contradictions found: {len(contra["contradictions"]) if contra else 0}')
if contra:
    for c in contra["contradictions"]:
        print(f'    Type: {c["type"]}, Severity: {c["severity"]}, Detail: {c["detail"]}')

print()
print("=" * 60)
print("TEST 4: Batch Contradiction Detection")
print("=" * 60)
steps = [
    {'action': 'scan', 'output': 'found bug', 'success': True},
    {'action': 'fix', 'output': 'file not found', 'success': False},
    {'action': 'test', 'expected_output': '成功通过', 'output': '失败错误异常', 'success': False},
    {'action': 'deploy', 'resource': 'config.yaml', 'action_type': 'write', 'success': True},
    {'action': 'update', 'resource': 'config.yaml', 'action_type': 'write', 'success': False},
]
batch = dcr.detect_contradictions_batch(steps)
print(f'  Steps: {len(steps)}, Contradiction pairs: {len(batch)}')
for b in batch:
    print(f'    Steps {b["step_indices"]}: {len(b["contradictions"])} contradictions')
    for c in b["contradictions"]:
        print(f'      {c["type"]} ({c["severity"]}): {c["detail"]}')

print()
print("=" * 60)
print("TEST 5: Chain Validation")
print("=" * 60)
validation = dcr.validate_chain(chain)
print(f'  Valid: {validation["valid"]}')
print(f'  Issues: {len(validation["issues"])}')
for issue in validation["issues"]:
    print(f'    {issue["type"]}: {issue}')

print()
print("=" * 60)
print("TEST 6: Causal Graph")
print("=" * 60)
print(f'  Graph size: {dcr.causal_graph.size()}')
print(f'  Has cycle: {dcr.causal_graph.has_cycle()}')
roots = dcr.causal_graph.get_roots()
leaves = dcr.causal_graph.get_leaves()
print(f'  Roots: {len(roots)}, Leaves: {len(leaves)}')
if roots:
    print(f'  Root event: {roots[0].event[:50]}')

print()
print("=" * 60)
print("TEST 7: Plan Steps Analysis")
print("=" * 60)
plan_steps = [
    {'action': '扫描代码', 'success': True, 'output': '发现3个问题'},
    {'action': '修复问题', 'success': True, 'output': '已修复'},
    {'action': '运行测试', 'success': False, 'output': '测试失败错误'},
    {'action': '部署', 'resource': 'app.py', 'action_type': 'write', 'success': True},
]
analysis = dcr.analyze_plan_steps('代码修复任务', plan_steps)
print(f'  Total steps: {analysis["total_steps"]}')
print(f'  Chain length: {analysis["chain_length"]}')
print(f'  Contradictions: {analysis["contradictions_found"]}')
print(f'  High severity: {analysis["high_severity_contradictions"]}')
print(f'  Chain valid: {analysis["chain_valid"]}')
print(f'  Recommendations: {analysis["recommendations"]}')

print()
print("=" * 60)
print("TEST 8: Similarity Computation")
print("=" * 60)
sim1 = dcr._compute_similarity("修复代码bug", "修复代码缺陷")
sim2 = dcr._compute_similarity("修复代码bug", "部署服务器")
sim3 = dcr._compute_similarity("refactor code", "refactor module")
print(f'  Similar(修复代码bug, 修复代码缺陷): {sim1}')
print(f'  Different(修复代码bug, 部署服务器): {sim2}')
print(f'  English(refactor code, refactor module): {sim3}')

print()
print("=" * 60)
print("TEST 9: Chain Merge")
print("=" * 60)
chain2 = dcr.trace_causal_chain('性能优化', [
    {'detail': '分析性能瓶颈'},
    {'detail': '优化数据库查询'},
    {'detail': '验证优化效果'},
])
merged = dcr.merge_chains(chain, chain2)
print(f'  Chain A: {len(chain)} steps, Chain B: {len(chain2)} steps')
print(f'  Merged: {len(merged)} steps')

print()
print("=" * 60)
print("TEST 10: Intervention")
print("=" * 60)
if len(chain) >= 2:
    target = chain[len(chain) // 2]
    intervention = dcr.intervene(chain, target.id, '增加代码审查步骤')
    print(f'  Intervention: {intervention["intervention"]}')
    print(f'  Position: {intervention["position"]}')
    print(f'  Downstream affected: {intervention["downstream_affected"]}')
    print(f'  Confidence: {intervention["confidence"]}')

print()
print("=" * 60)
print("TEST 11: Stats")
print("=" * 60)
stats = dcr.get_stats()
for k, v in stats.items():
    if k != 'recent_contradictions':
        print(f'  {k}: {v}')

print()
print("=" * 60)
print("TEST 12: Causal Summary")
print("=" * 60)
summary = dcr.generate_causal_summary('代码修复', chain)
print(summary[:300])

print()
print("ALL BASIC TESTS COMPLETED!")
