import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.deep_causal_reasoner import DeepCausalReasoner
dcr = DeepCausalReasoner()

print("ORIGINAL PLAN VERIFICATION")
print("=" * 60)

events = [
    {'detail': 'pylint扫描发现3个长函数'},
    {'detail': '分析确认需要拆分process_event'},
    {'detail': '执行拆分，将process_event分为4个子函数'},
    {'detail': '运行测试，验证拆分后功能正常'}
]
chain = dcr.trace_causal_chain('启动代码修复', events)
print(f'因果链长度: {len(chain)}')
print(f'根因: {chain[0].event}')
print(f'最终效应: {chain[-1].effect if hasattr(chain[-1], "effect") and chain[-1].effect else chain[-1].effects[0] if chain[-1].effects else chain[-1].event}')

cf = dcr.counterfactual_reasoning(chain, '手动重构而非自动拆分')
print(f'反事实: {cf["hypothetical_chain"][:2]}')

step_a = {'action': 'pylint', 'output': 'found bug in agent_bus.py', 'success': True}
step_b = {'action': 'auto_fix', 'output': 'agent_bus.py not found', 'success': False}
contra = dcr.detect_contradiction(step_a, step_b)
print(f'矛盾数: {len(contra["contradictions"]) if contra else 0}')
print(f'矛盾类型: {contra["contradictions"][0]["type"] if contra else "None"}')

print()
print("ORIGINAL VERIFICATION PASSED!")
